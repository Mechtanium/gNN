# pylint: disable=too-many-lines
"""Field class."""
import hashlib
import logging
import os
import sys
import tempfile
import weakref
import zipfile
from copy import deepcopy
from functools import partial
from string import Template
from anytree import PreOrderIter

import h5py
import numpy as np
import pandas as pd

from .arithmetics import load_add, load_copy, load_equals, load_multiply
from .faults import Faults
from .aquifer import Aquifers
from .configs import default_config
from .grids import CornerPointGrid, Grid, OrthogonalGrid, specify_grid
from .parse_utils import (dates_to_str, preprocess_path,
                          read_dates_from_buffer, tnav_ascii_parser)
from .rock import Rock
from .states import States
from .tables import Tables
from .template_models import (CORNERPOINT_GRID, DEFAULT_ECL_MODEL,
                              DEFAULT_TN_MODEL, ORTHOGONAL_GRID)
from .utils import get_single_path
from .validation import InvalidModelDeckError, classify_model_deck
from .wells import Wells

ACTOR = None

COMPONENTS_DICT = {'cornerpointgrid': ['grid', CornerPointGrid],
                   'orthogonalgrid': ['grid', OrthogonalGrid],
                   'grid': ['grid', Grid],
                   'rock': ['rock', Rock],
                   'states': ['states', States],
                   'wells': ['wells', Wells],
                   'tables': ['tables', Tables],
                   'aquifers': ['aquifers', Aquifers],
                   'faults': ['faults', Faults]
                   }

DEFAULT_HUNITS = {'METRIC': ['sm3/day', 'ksm3/day', 'ksm3', 'Msm3', 'bara'],
                  'FIELD': ['stb/day', 'Mscf/day', 'Mstb', 'MMscf', 'psia']}

SECTIONS_DICT = {
    'GRID': [('PORO', 'rock'), ('PERMX', 'rock'), ('PERMY', 'rock'), ('PERMZ', 'rock'), ('MULTZ', 'rock')],
    'PROPS': [('SWATINIT', 'rock'), ('SWL', 'rock'), ('SWCR', 'rock'), ('SGU', 'rock'), ('SGL', 'rock'),
              ('SGCR', 'rock'), ('SOWCR', 'rock'), ('SOGCR', 'rock'), ('SWU', 'rock'), ('ISWCR', 'rock'),
              ('ISGU', 'rock'), ('ISGL', 'rock'), ('ISGCR', 'rock'), ('ISWU', 'rock'), ('ISGU', 'rock'),
              ('ISGL', 'rock'), ('ISWL', 'rock'), ('ISOGCR', 'rock'), ('ISOWCR', 'rock')]
}

SUMMARY_KW = ['WLPR', 'WOPR', 'WGPR', 'WWPR', 'WWIR', 'WGIR', 'WBHP',
              'EXCEL', 'RPTONLY', 'SEPARATE']

META_KW = ['ARRA', 'ARRAY', 'DATES', 'TITLE', 'START', 'METRIC', 'FIELD',
           'HUNI', 'HUNITS', 'OIL', 'GAS', 'WATER', 'DISGAS', 'VAPOIL', 'RES',
           'RESTARTDATE', 'RESTART'] + SUMMARY_KW


#pylint: disable=protected-access
class FieldState:
    """State holder."""
    def __init__(self, field):
        self._field = weakref.ref(field)

    @property
    def field(self):
        """Reference Field."""
        return self._field()


class Field:
    """Reservoir model.

    Contains components of the reservoir model and preprocessing tools.

    Parameters
    ----------
    path : str, optional
        Path to source model files.
    config : dict, optional
        Components and attributes to load.
    logfile : str, optional
        Path to log file.
    encoding : str, optional
        Files encoding. Set 'auto' to infer encoding from initial file block.
        Sometimes it might help to specify block size, e.g. 'auto:3000' will
        read first 3000 bytes to infer encoding.
    loglevel : str, optional
        Log level to be printed while loading. Default to 'INFO'.
    """
    _default_config = default_config
    def __init__(self, path=None, config=None, logfile=None, encoding='auto', loglevel='INFO'):
        self._path = preprocess_path(path) if path is not None else None
        self._encoding = encoding
        self._logfile = logfile
        self._loglevel = loglevel
        self._source_config = deepcopy(config)
        self._components = {}
        self._config = None
        self._meta = {'UNITS': 'METRIC',
                      'START': pd.to_datetime(''),
                      'DATES': pd.to_datetime([]),
                      'FLUIDS': [],
                      'SUMMARY': [],
                      'MODEL_TYPE': '',
                      'HUNITS': DEFAULT_HUNITS['METRIC']}
        self._state = FieldState(self)

        logging.shutdown()
        handlers = [logging.StreamHandler(sys.stdout)]
        if logfile is not None:
            handlers.append(logging.FileHandler(logfile, mode='w'))
        logging.basicConfig(handlers=handlers)
        self._logger = logging.getLogger('Field')
        self._logger.setLevel(getattr(logging, loglevel))

        if self._path is not None:
            self._init_components(config)

        self._pyvista_grid = None

    def _init_components(self, config):
        """Initialize components."""
        fmt = self._path.suffix.strip('.').upper()
        if config is not None:
            config = {k.lower(): self._config_parser(v) for k, v in config.items()}
        if fmt == 'HDF5':
            with h5py.File(self._path, 'r') as f:
                keys = [k.lower() for k in f]
                if config is None:
                    config = {k: {'attrs': None, 'kwargs': {}} for k in keys}
                elif 'grid' in config:
                    if 'cornerpointgrid' in keys:
                        config['cornerpointgrid'] = config.pop('grid')
                    elif 'orthogonalgrid' in keys:
                        config['orthogonalgrid'] = config.pop('grid')
        elif config is None:
            self._logger.info('Using default config.')
            config = {k.lower(): self._config_parser(v) for k, v in default_config.items()}

        for k in config:
            setattr(self, COMPONENTS_DICT[k][0], COMPONENTS_DICT[k][1](field=self))
        self._config = {COMPONENTS_DICT[k][0]: v for k, v in config.items()}

    @staticmethod
    def _config_parser(value):
        """Separate config into attrs and kwargs."""
        if isinstance(value, str):
            attrs = [value.upper()]
            kwargs = {}
        elif isinstance(value, (list, tuple)):
            attrs = [x.upper() for x in value]
            kwargs = {}
        elif isinstance(value, dict):
            attrs = value['attrs']
            if attrs is None:
                pass
            elif isinstance(attrs, str):
                attrs = [attrs.upper()]
            else:
                attrs = [x.upper() for x in attrs]
            kwargs = {k: v for k, v in value.items() if k != 'attrs'}
        else:
            raise TypeError("Component's config should be of type str, list, tuple or dict. Found {}."
                            .format(type(value)))
        return {'attrs': attrs, 'kwargs': kwargs}

    @property
    def meta(self):
        """"Model meta data."""
        return self._meta

    @property
    def state(self):
        """"Field state."""
        return self._state

    @property
    def start(self):
        """Model start time in a datetime format."""
        return pd.to_datetime(self.meta['START'])

    @property
    def path(self):
        """Path to original model."""
        if self._path is not None:
            return str(self._path)
        raise ValueError("Model has no file to originate from.")

    @property
    def basename(self):
        """Model filename without extention."""
        fname = os.path.basename(self.path)
        return os.path.splitext(fname)[0]

    @property
    def components(self):
        """Model components."""
        return tuple(self._components.keys())

    def items(self):
        """Returns pairs of components's names and instance."""
        return self._components.items()

    @property
    def grid(self):
        """Grid component."""
        return self._components['grid']

    @grid.setter
    def grid(self, x):
        """Grid component setter."""
        x.field = self
        self._components['grid'] = x
        return self

    @property
    def wells(self):
        """Wells component."""
        return self._components['wells']

    @wells.setter
    def wells(self, x):
        """Wells component setter."""
        x.field = self
        self._components['wells'] = x
        return self

    @property
    def rock(self):
        """Rock component."""
        return self._components['rock']

    @rock.setter
    def rock(self, x):
        """Rock component setter."""
        x.field = self
        self._components['rock'] = x
        return self

    @property
    def states(self):
        """States component."""
        return self._components['states']

    @states.setter
    def states(self, x):
        """States component setter."""
        x.field = self
        self._components['states'] = x
        return self

    @property
    def faults(self):
        """Faults component."""
        return self._components['faults']

    @faults.setter
    def faults(self, x):
        """Faults component setter."""
        x.field = self
        self._components['faults'] = x
        return self

    @property
    def aquifers(self):
        """Aquifers component."""
        return self._components['aquifers']

    @aquifers.setter
    def aquifers(self, x):
        """States component setter."""
        x.field = self
        self._components['aquifers'] = x
        return self

    @property
    def tables(self):
        """Tables component."""
        return self._components['tables']

    @tables.setter
    def tables(self, x):
        """Tables component setter."""
        x.field = self
        self._components['tables'] = x
        return self

    @property
    def result_dates(self):
        """Result dates, actual if present, target otherwise."""
        return self.wells.result_dates

    def set_state(self, **kwargs):
        """State setter."""
        for k, v in kwargs.items():
            setattr(self.state, k, v)
        return self

    def copy(self):
        """Returns a deepcopy of Field."""
        copy = self.__class__()
        for k, v in self.items():
            setattr(copy, k, v.copy())
        copy._meta = deepcopy(self.meta) #pylint: disable=protected-access
        return copy

    def load(self, raise_errors=False, include_binary=True):
        """Load model components.

        Parameters
        ----------
        raise_errors : bool
            Error handling mode. If True, errors will be raised and stop loading.
            If False, errors will be printed but do not stop loading.
        include_binary : bool
            Read data from binary files in RESULTS folder. Default to True.

        Returns
        -------
        out : Field
            Field with loaded components.
        """
        name = os.path.basename(self._path)
        fmt = os.path.splitext(name)[1].strip('.')

        if fmt.upper() == 'HDF5':
            self._load_hdf5(raise_errors=raise_errors)
        elif fmt.upper() in ['DATA', 'DAT']:
            deck_kind, root_keywords = classify_model_deck(self._path, encoding=self._encoding)
            if (deck_kind == 'restart') and (type(self) is Field):
                return self._load_restart_field(raise_errors=raise_errors, include_binary=include_binary)
            if deck_kind == 'invalid_fragment':
                raise InvalidModelDeckError(
                    f"'{self.path}' is not a standalone model deck. "
                    f"Expected the root file to declare one of {', '.join(['RUNSPEC', 'RESTART', 'RESTARTDATE'])}; "
                    f"found root keywords {list(root_keywords[:8])}."
                )
            self._load_data(raise_errors=raise_errors, include_binary=include_binary)
        else:
            raise NotImplementedError('Format {} is not supported.'.format(fmt))

        self._collect_loaded_attrs()

        return self

    def _load_restart_field(self, raise_errors=False, include_binary=True):
        """Dispatch restart decks to RestartField while preserving init settings."""
        from .restart_field import RestartField

        restart_field = RestartField(
            path=self.path,
            config=deepcopy(self._source_config),
            logfile=self._logfile,
            encoding=self._encoding,
            loglevel=self._loglevel,
        )
        return restart_field.load(raise_errors=raise_errors, include_binary=include_binary)

    def _load_hdf5(self, raise_errors):
        """Load model in HDF5 format."""
        with h5py.File(self.path, 'r') as f:
            for k, v in f.attrs.items():
                if k == 'DATES':
                    self.meta['DATES'] = pd.to_datetime(v)
                else:
                    self.meta[k] = v
        for comp, config in self._config.items():
            getattr(self, comp).load(self.path,
                                     attrs=config['attrs'],
                                     raise_errors=raise_errors,
                                     logger=self._logger,
                                     **config['kwargs'])
        return self

    def _get_results_path(self, raise_errors):
        """Return a path to extracted RESULTS data if available."""
        model_dir = os.path.dirname(self.path)
        path_to_results = os.path.join(model_dir, 'RESULTS')
        if os.path.exists(path_to_results):
            return path_to_results

        path_to_results_zip = path_to_results + '.zip'
        if os.path.exists(path_to_results_zip):
            cache_key = hashlib.md5(os.path.abspath(path_to_results_zip).encode('utf-8')).hexdigest()
            cache_root = os.path.join(tempfile.gettempdir(), 'deepfield_results', cache_key)
            extracted_results = os.path.join(cache_root, 'RESULTS')
            if not os.path.exists(extracted_results):
                os.makedirs(cache_root, exist_ok=True)
                self._logger.info('Extracting %s to %s.', path_to_results_zip, extracted_results)
                with zipfile.ZipFile(path_to_results_zip) as archive:
                    archive.extractall(cache_root)
            return extracted_results

        if raise_errors:
            raise ValueError("RESULTS folder or RESULTS.zip was not found in model directory.")
        self._logger.warning("RESULTS folder or RESULTS.zip was not found in model directory.")
        return None

    def _load_binary(self, components, raise_errors):
        """Load data from binary files in RESULTS folder or RESULTS.zip archive."""
        path_to_results = self._get_results_path(raise_errors=raise_errors)
        if path_to_results is None:
            return
        for comp in components:
            if comp in self._config:
                getattr(self, comp).load(path_to_results,
                                         attrs=self._config[comp]['attrs'],
                                         basename=self.basename,
                                         logger=self._logger,
                                         **self._config[comp]['kwargs'])

    def _get_loaders(self, config):
        loaders = {}
        for k in META_KW:
            loaders[k] = partial(self._read_buffer, attr=k, logger=self._logger)

        loaders['COPY'] = partial(load_copy, self, logger=self._logger)
        loaders['MULTIPLY'] = partial(load_multiply, self, logger=self._logger)
        loaders['EQUALS'] = partial(load_equals, self, logger=self._logger)
        loaders['ADD'] = partial(load_add, self, logger=self._logger)
        for comp, conf in config.items():
            if conf['attrs'] is not None:
                attrs = list(set(conf['attrs']) - set(getattr(self, comp).state.binary_attributes))
            else:
                attrs = None
            kwargs = conf['kwargs']
            if comp in ['grid', 'rock', 'states', 'tables', 'faults']:
                assert attrs is not None
                for k in attrs:
                    loaders[k] = partial(getattr(self, comp).load, attr=k,
                                         logger=self._logger, **kwargs)
            if comp == 'wells':
                extented_list = []
                assert attrs is not None
                for k in attrs:
                    if k in ['PERF', 'EVENTS']:
                        extented_list.extend(['EFIL', 'EFILE', 'ETAB'])
                    elif k == 'HISTORY':
                        extented_list.extend(['HFIL', 'HFILE'])
                    elif k == 'WELLTRACK':
                        extented_list.extend(['TFIL', 'WELLTRACK'])
                    elif k == 'RESULTS':
                        continue
                    else:
                        extented_list.append(k)
                if kwargs.get('groups', True):
                    extented_list.extend(['GROU', 'GROUP', 'GRUPTREE'])

                for k in set(extented_list):
                    loaders[k] = partial(self.wells.load, attr=k, logger=self._logger,
                                         meta=self.meta, grid=self.grid, **kwargs)
            if comp == 'aquifers':
                for k in ['AQCT', 'AQCO', 'AQUANCON', 'AQUCT']:
                    loaders[k] = partial(self.aquifers.load, attr=k, logger=self._logger)
        return loaders

    def _load_results(self, raise_errors, include_binary):
        config = self._config
        if (('wells' in config) and ('attrs' in config['wells']) and
            ('RESULTS' in config['wells']['attrs'])):
            path_to_results = self._get_results_path(raise_errors=raise_errors)
            if path_to_results is None:
                return self
            rsm = get_single_path(path_to_results, self.basename + '.RSM', self._logger)
            if rsm is None and not include_binary:
                if raise_errors:
                    raise ValueError("RSM file was not found in model directory.")
                self._logger.warning("RSM file was not found in model directory.")
            if rsm is not None and 'RESULTS' not in self.wells.state.binary_attributes:
                self.wells.load(rsm, logger=self._logger)
        return self

    def _check_vapoil(self):
        if 'VAPOIL' in self.meta['FLUIDS'] and 'tables' in self.components:
            # TODO should we make a kwarg for convertion key?
            self.tables.pvtg_to_pvdg(as_saturated=False)
            self.meta['FLUIDS'].remove('VAPOIL')
            self._logger.warning(
                """Vaporized oil option is not currently supported.
                PVTG table is converted into PVDG one."""
            )
        return self

    def _load_data(self, raise_errors=False, include_binary=True):
        """Load model in DATA format."""
        if include_binary:
            self._load_binary(components=('grid',),
                              raise_errors=raise_errors)
            if 'ACTNUM' in self.grid.state.binary_attributes:
                self._load_binary(components=('rock',), raise_errors=raise_errors)

        loaders = self._get_loaders(self._config)
        tnav_ascii_parser(self._path, loaders, self._logger, encoding=self._encoding,
                          raise_errors=raise_errors)

        self.grid = specify_grid(self.grid)

        if 'MINPV' in self.grid.attributes:
            if 'ACTNUM' in self.grid.state.binary_attributes:
                self._logger.info('ACTNUM is loaded from binary file: MINPV was not applied.')
            else:
                self.grid.apply_minpv()
                self._logger.info('MINPV {} is applied.'.format(self.grid.minpv[0]))

        if include_binary:
            self._load_binary(components=('states', 'wells'), raise_errors=raise_errors)

        self._load_results(raise_errors, include_binary)
        self._check_vapoil()

        if 'wells' in self.components:
            self.wells.add_welltrack()
            for well in self.wells:
                if 'COMPDAT' in well or 'COMPDATL' in well:
                    self.meta['MODEL_TYPE'] = 'ECL'
                    break
            else:
                self.meta['MODEL_TYPE'] = 'TN'
            self._logger.info('Model type is determined as {}.'.format(self.meta['MODEL_TYPE']))


        if self._config['grid']['kwargs'].get('apply_mapaxes', False):
            self.grid.map_grid()
            self._logger.info('Grid pillars `COORD` are mapped to new axis with respect to `MAPAXES`.')

        if 'states' in self.components:
            if not self.states.state.binary_attributes and self.states.attributes:
                self.states.dates = pd.to_datetime([self.meta['START']])
                self._logger.info('States dates are set to start date {}.'.format(self.meta['START']))

        return self

    def _read_buffer(self, buffer, attr, logger):
        """Load model meta attributes."""
        if attr in ['TITLE', 'START']:
            self.meta[attr] = next(buffer).split('/')[0].strip(' \t\n\'\""')
        elif attr == 'DATES':
            date = pd.to_datetime(next(buffer).split('/')[:1])
            self.meta['DATES'] = self.meta['DATES'].append(date)
        elif attr in ['ARRA', 'ARRAY']:
            dates = read_dates_from_buffer(buffer, attr, logger)
            self.meta['DATES'] = self.meta['DATES'].append(dates)
        elif attr in ['METRIC', 'FIELD']:
            self.meta['UNITS'] = attr
        elif attr in ['HUNI', 'HUNITS']:
            self._read_hunits(next(buffer))
        elif attr in ['OIL', 'GAS', 'WATER', 'DISGAS', 'VAPOIL']:
            self.meta['FLUIDS'].append(attr)
        elif attr in SUMMARY_KW:
            self.meta['SUMMARY'].append(attr)
        else:
            raise NotImplementedError("Keyword {} is not supported.".format(attr))
        return self

    def _read_hunits(self, line):
        """Parse HUNIts from line."""
        units = line.strip('/\t\n ').split()
        self.meta['HUNITS'] = []
        defaults = DEFAULT_HUNITS[self.meta['UNITS']]
        for k in units:
            if '*' in k:
                n = int(k[0])
                nread = len(self.meta['HUNITS'])
                self.meta['HUNITS'].extend(defaults[nread: nread + n])
            else:
                self.meta['HUNITS'].append(k)
        assert len(self.meta['HUNITS']) == len(defaults), 'Missmatch of HUNITS array length'
        return self

    def _collect_loaded_attrs(self):
        """Collect loaded attributes."""
        out = {}
        self._logger.info("===== Field summary =====")
        for comp in self.components:
            if comp == 'wells':
                attrs = []
                for node in PreOrderIter(self.wells.root):
                    attrs.extend(list(node.attributes))
                attrs = list(set(attrs))
            elif comp == 'faults':
                attrs = []
                for node in PreOrderIter(self.faults.root):
                    attrs.extend(list(node.attributes))
                attrs = list(set(attrs))
            elif comp == 'aquifers':
                attrs = []
                for _, aqf in self.aquifers.items():
                    attrs.extend(list(aqf.attributes))
                attrs = list(set(attrs))
            else:
                attrs = getattr(self, comp).attributes
            msg = "{} attributes: {}".format(comp.upper(), ', '.join(attrs))
            out[comp.upper()] = attrs
            self._logger.info(msg)
        self._logger.info("=========================")
        return out

    def _dump_hdf5(self, path, mode='w', only_active=False, reduce_floats=True, **kwargs):
        """Dump model into HDF5 file.

        Parameters
        ----------
        path : str
            Path to output file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.
        only_active : bool
            Keep state values in active cells only. Default to False.
        reduce_floats : bool
            if True, float precision will be reduced to np.float32 to save disk space.
        kwargs : misc
            Any additional named arguments to component's ``dump``.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        float_precision = {
            'grid': np.float32 if reduce_floats else None,
            'rock': np.float32 if reduce_floats else None,
            'states': np.float32 if reduce_floats else None,
        }
        with h5py.File(path, mode) as f:
            for k, v in self.meta.items():
                if k == 'DATES':
                    f.attrs['DATES'] = pd.to_datetime(self.meta['DATES']).astype(np.int64)
                else:
                    f.attrs[k] = v
        for k, comp in self.items():
            if k == 'states':
                comp.dump(path=path, mode='a', float_dtype=float_precision.get(k, None),
                          actnum=self.grid.actnum if only_active else None, **kwargs)
            else:
                comp.dump(path=path, mode='a', float_dtype=float_precision.get(k, None), **kwargs)
        return self

    def _dump_ascii(self, dir_path, title, **kwargs):
        """Dump model's data files in tNav project.

        Parameters
        ----------
        dir_path : str
            Directory where model files will be placed.
        title : str
            Model title.
        kwargs : misc
            Any additional named arguments that will be substituted to model template file.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        dir_inc = os.path.join(dir_path, 'INCLUDE')
        if not os.path.exists(dir_inc):
            os.mkdir(dir_inc)
        datafile = os.path.join(dir_path, title + '.data')

        model_type = self.meta.get('MODEL_TYPE', 'TN')
        if model_type == 'TN':
            template = Template(DEFAULT_TN_MODEL)
        elif model_type == 'ECL':
            template = Template(DEFAULT_ECL_MODEL)
        else:
            raise ValueError('Unknown model type {}'.format(model_type))

        fill_values = {'title': title, 'units': self.meta['UNITS']}
        fill_values['phases'] = '\n'.join(self.meta['FLUIDS'])

        if 'START' in self.meta:
            fill_values['start'] = self.meta['START']

        fill_values['dates'] = dates_to_str(self.result_dates)

        fill_values['dimens'] = ' '.join(self.grid.dimens.astype(str))
        fill_values['size'] = np.prod(self.grid.dimens)

        def compressed_str(arr):
            "Compressed array string."
            out = ''
            d = np.hstack([[0], np.where(np.diff(arr) != 0)[0]+1, [len(arr)]])
            for i in range(len(d)-1):
                val = arr[d[i]]
                count = d[i+1] - d[i]
                if count > 1:
                    out += '{}*{} '.format(count, val)
                else:
                    out += '{} '.format(val)
            return out + '/'

        if isinstance(self.grid, OrthogonalGrid):
            template = Template(template.safe_substitute(grid_specs=ORTHOGONAL_GRID))
            fill_values.update(dict(
                mapaxes=' '.join(self.grid.mapaxes.astype(str)),
                dx=compressed_str(self.grid.ravel('dx')),
                dy=compressed_str(self.grid.ravel('dy')),
                dz=compressed_str(self.grid.ravel('dz')),
                tops=compressed_str(self.grid.ravel('tops'))
            ))
        elif isinstance(self.grid, CornerPointGrid):
            template = Template(template.safe_substitute(grid_specs=CORNERPOINT_GRID))
            coord = os.path.join('INCLUDE', 'coord.inc')
            self.grid.dump(os.path.join(dir_path, coord), attrs='COORD', compressed=False)
            zcorn = os.path.join('INCLUDE', 'zcorn.inc')
            self.grid.dump(os.path.join(dir_path, zcorn), attrs='ZCORN')
            fill_values.update({'coord': coord, 'zcorn': zcorn})
        else:
            raise NotImplementedError("Dump for grid of type {} is not implemented."
                                      .format(self.grid.__class__.__name__))

        inc = os.path.join('INCLUDE', 'actnum.inc')
        self.grid.dump(os.path.join(dir_path, inc), attrs='ACTNUM', fmt='%i')
        fill_values['actnum'] = inc

        tmp = ''
        for attr_name, comp_name in SECTIONS_DICT['GRID']:
            if attr_name in getattr(self, comp_name).attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr_name.lower())
                inc = os.path.join('INCLUDE', attr_name.lower() + '.inc')
                getattr(self, comp_name).dump(os.path.join(dir_path, inc), attrs=attr_name, fmt='%.3f')
                fill_values[attr_name.lower()] = inc
        template = Template(template.safe_substitute(rock_grid=tmp))

        if 'faults' in self.components:
            attrs = ['faults', 'multflt']
            for attr in attrs:
                inc = os.path.join('INCLUDE', attr + '.inc')
                self.faults.dump(os.path.join(dir_path, inc), attr=attr)
                fill_values[attr] = inc

        tmp = ''
        if 'aquifers' in self.components:
            tmp += "INCLUDE\n'${} '/\n/\n\n".format('aquifers_file')
            inc = os.path.join('INCLUDE', 'aquifers.inc')
            self.aquifers.dump(os.path.join(dir_path, inc))
            fill_values['aquifers_file'] = inc
        template = Template(template.safe_substitute(aquifers=tmp))

        if model_type == 'TN':
            attrs = ['welltrack', 'perf', 'group', 'events']
        elif model_type == 'ECL':
            attrs = ['gruptree', 'schedule', 'welspecs']
        else:
            raise ValueError(f'Model type {model_type} is not supported.')

        for attr in attrs:
            inc = os.path.join('INCLUDE', attr + '.inc')
            self.wells.dump(os.path.join(dir_path, inc), attr=attr, grid=self.grid,
                            dates=self.result_dates, start_date=self.start)
            fill_values[attr] = inc

        tmp = ''
        if 'states' in self.components:
            for attr in self.states.attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr.lower())
                inc = os.path.join('INCLUDE', attr.lower() + '.inc')
                self.states.dump(os.path.join(dir_path, inc), attrs=attr, fmt='%.3f')
                fill_values[attr.lower()] = inc
        template = Template(template.safe_substitute(states=tmp))

        tmp = ''
        if 'tables' in self.components:
            for attr in self.tables.attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr.lower())
                inc = os.path.join('INCLUDE', attr.lower() + '.inc')
                self.tables.dump(os.path.join(dir_path, inc), attrs=attr)
                fill_values[attr.lower()] = inc
        template = Template(template.safe_substitute(tables=tmp))
        tmp = ''
        for attr_name, comp_name in SECTIONS_DICT['PROPS']:
            if attr_name in getattr(self, comp_name).attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr_name.lower())
                inc = os.path.join('INCLUDE', attr_name.lower() + '.inc')
                getattr(self, comp_name).dump(os.path.join(dir_path, inc), attrs=attr_name, fmt='%.3f')
                fill_values[attr_name.lower()] = inc
        template = Template(template.safe_substitute(rock_props=tmp))
        template = Template(template.safe_substitute(fill_values))
        template = Template(template.safe_substitute(kwargs))

        out = template.safe_substitute()
        missing = Template.pattern.findall(out)
        if missing:
            self._logger.warning('Dump missed values for %s.', ', '.join([i[1] for i in missing]))

        with open(datafile, 'w') as f:
            f.writelines(out)
        return self

    def _dump_binary_results(self, dir_path, mode, title):
        """Dump model's binary result files.

        Parameters
        ----------
        dir_path : str
            Path to location where RESULTS directory will be created.
        title : str
            Model title.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        dir_res = os.path.join(dir_path, 'RESULTS')
        if not os.path.exists(dir_res):
            os.mkdir(dir_res)

        grid_dim = self.grid.dimens
        time_size = self.states.n_timesteps

        dir_name = os.path.join(dir_res, title)

        units_type = {
            'METRIC': 1,
            'FIELD': 2,
            'LAB': 3,
            'PVT-M': 4,
        }[self.meta['UNITS']]

        grid_type = {
            'CornerPointGrid': 0,
            'OrthogonalGrid': 3,
        }[self.grid.class_name]

        grid_format = {
            'CornerPointGrid': 1,
            'OrthogonalGrid': 2,
        }.get(self.grid.class_name, 0)# 0 - Unknown; 1 - Corner point; 2 - Block centered

        i_phase = 0
        for elem in self.meta['FLUIDS']:
            i_phase += {'OIL': 1, 'WATER': 2, 'GAS': 4}.get(elem, 0)

        egrid.save_egrid(self.grid.as_corner_point, dir_name, grid_dim, grid_format, mode)

        init.save_init(self.rock, dir_name, grid_dim, self.grid.actnum.sum(),
                       units_type, grid_type, self.start, i_phase, mode)

        is_unified = True

        restart.save_restart(is_unified,
                             dir_name,
                             self.states.strip_na(),
                             self.states.attributes,
                             self.states.dates,
                             grid_dim,
                             time_size,
                             mode,
                             self._logger)

        rates = {}
        for well in self.wells.main_branches:
            curr_rates = self.wells[well].total_rates
            well_data = {}
            for k in ['WWPR', 'WOPR', 'WGPR']:
                if k in curr_rates:
                    well_data[k.lower()] = curr_rates[k].values.astype('float64')
            if well_data:
                rates[well] = well_data

        summary.save_summary(is_unified, dir_name, rates, self.result_dates,
                             grid_dim, mode, self._logger)
        return self

    def history_to_results(self):
        """Convert history to results."""
        for node in self.wells:
            if node.is_main_branch and not hasattr(node, 'history'):
                self.wells.drop(node.name)

        for node in self.wells:
            if hasattr(node, 'results'):
                delattr(node, 'results')

        rename = {'QOIL': 'WOPR', 'QGAS': 'WGPR', 'QWAT': 'WWPR', 'QWIN': 'WWIR', 'BHP': 'WBHP'}
        for node in self.wells:
            if node.is_main_branch:
                new_results = node.history.rename(columns=rename)[['DATE'] + list(rename.values())]
                node.results = new_results.drop_duplicates(subset='DATE')
                node.results = node.results.reset_index(drop=True)
        return self

    # pylint: disable=protected-access
