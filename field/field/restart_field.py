"""RestartField class."""
import copy
from functools import partial

import pandas as pd
import numpy as np

from .configs import default_config_restart
from .field import Field
from .parse_utils import read_restartdate_from_buffer, tnav_ascii_parser
from .parse_utils.ascii import _get_path, read_restart_from_buffer
from .states import States

class RestartField(Field):
    """
    Restart Reservoir Model.

    Incorporate data and provide routines to work with restaart models.

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

    _default_config = default_config_restart

    def __init__(self, path=None, config=None, logfile=None, encoding='auto', loglevel='INFO'):
        super().__init__(path, config, logfile, encoding, loglevel)
        self.parent = None

    def _init_components(self, config):
        config = self._ensure_restart_config(config)
        if config is None:
            config = self._default_config

        config_tmp = {k.lower(): self._config_parser(v) for k, v in config.items() if k.lower()!='parentmodel'}
        config_tmp['ParentModel'] = config['ParentModel']
        config = config_tmp
        parent_model_config = config['ParentModel']
        restart_config = copy.deepcopy(config)
        del restart_config['ParentModel']
        super()._init_components(restart_config)
        self._restart_config = restart_config
        self._config = config
        self._parent_model_config = parent_model_config

    def _ensure_restart_config(self, config):
        if config is None:
            return None
        parent_model_config = config.get('ParentModel', config.get('parentmodel'))
        if parent_model_config is not None:
            canonical_config = {
                k: copy.deepcopy(v) for k, v in config.items() if k.lower() != 'parentmodel'
            }
            canonical_config['ParentModel'] = copy.deepcopy(parent_model_config)
            return canonical_config
        states_config = copy.deepcopy(config.get('States', config.get('states', self._default_config['States'])))
        wells_config = copy.deepcopy(config.get('Wells', config.get('wells', self._default_config['Wells'])))
        return {'States': states_config, 'Wells': wells_config, 'ParentModel': copy.deepcopy(config)}

    @property
    def restart_date(self):
        "Show restart date."
        if 'RESTART' in self.meta:
            step = None
            if len(self.meta['RESTART']) > 1 and self.meta['RESTART'][1] is not None:
                try:
                    step = int(self.meta['RESTART'][1])
                except (TypeError, ValueError):
                    step = None
            parent_dates = getattr(self.parent.states, 'dates', pd.to_datetime([]))
            if len(parent_dates):
                if step is None:
                    return parent_dates[0]
                idx = min(max(step - 1, 0), len(parent_dates) - 1)
                return parent_dates[idx]
            meta_dates = pd.to_datetime(self.parent.meta.get('DATES', []))
            if len(meta_dates):
                if step is None:
                    return meta_dates[0]
                idx = min(max(step - 1, 0), len(meta_dates) - 1)
                return meta_dates[idx]
            return pd.to_datetime(self.parent.meta.get('START'))
        restart_date = self.meta['RESTARTDATE']
        restart_date = pd.to_datetime(f'{restart_date[4]}-{restart_date[3]}' + f'{restart_date[2]}')
        parent_dates = getattr(self.parent.states, 'dates', pd.to_datetime([]))
        if len(parent_dates):
            eligible_dates = parent_dates[parent_dates <= restart_date]
            if len(eligible_dates):
                return eligible_dates[-1]
        return restart_date

    @property
    def components(self):
        if self.parent is None:
            return super().components
        components = list(self.parent.components)
        for comp in super().components:
            if comp not in components:
                components.append(comp)
        return tuple(components)

    def _parent_component(self, name):
        assert self.parent is not None
        return getattr(self.parent, name)

    @property
    def grid(self):
        return self._parent_component('grid')

    @property
    def rock(self):
        return self._parent_component('rock')

    @property
    def tables(self):
        return self._parent_component('tables')

    @property
    def aquifers(self):
        return self._parent_component('aquifers')

    @property
    def faults(self):
        return self._parent_component('faults')

    @property
    def restart_path(self):
        """Name of parent model.
        """
        if 'RESTART' in self._meta:
            return self._meta['RESTART'][0]
        return self._meta['RESTARTDATE'][0]


    def _load_data(self, raise_errors=False, include_binary=True):
        loaders = {
            'RESTARTDATE': partial(self._read_buffer, attr='RESTARTDATE', logger=self._logger),
            'RESTART': partial(self._read_buffer, attr='RESTART', logger=self._logger),
        }
        tnav_ascii_parser(self._path, loaders, self._logger, encoding=self._encoding,
                          raise_errors=raise_errors)
        assert self._path is not None
        data_dir = self._path.parent
        parent_reference = self.restart_path.strip('"\'')
        restart_path = parent_reference + '.data'
        try:
            parent_path = _get_path(restart_path, data_dir, self._logger, raise_errors=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not resolve restart parent model '{parent_reference}' from '{self.path}'."
            ) from exc
        if parent_path is None:
            raise FileNotFoundError(
                f"Could not resolve restart parent model '{parent_reference}' from '{self.path}'."
            )
        self.parent = Field(str(parent_path),
                            config=self._parent_model_config,
                            encoding=self._encoding,
                            loglevel=self._loglevel).load(
                                raise_errors=raise_errors,
                            )
        self._meta = {**self.parent.meta, **self.meta}
        self._meta['START'] = self.restart_date
        self.wells = self.parent.wells.copy()
        loaders = self._get_loaders(self._restart_config)
        tnav_ascii_parser(self._path, loaders, self._logger, encoding=self._encoding,
                          raise_errors=raise_errors)
        if include_binary:
            self._load_binary(components=('states', 'wells'), raise_errors=raise_errors)
        self._load_results(raise_errors, include_binary)
        self._check_vapoil()
        return self

    def _read_buffer(self, buffer, attr, logger):
        if attr not in ('RESTARTDATE', 'RESTART'):
            return super()._read_buffer(buffer, attr, logger)
        if attr == 'RESTARTDATE':
            self.meta['RESTARTDATE'] = read_restartdate_from_buffer(buffer, attr, logger)
        else:
            self.meta['RESTART'] = read_restart_from_buffer(buffer, attr, logger)
        return self

    def full_model(self):
        """Concatenate History Model and Restart Model.

        Returns
        -------
        model: Field
        Concatenated model.
        """
        tmp_model = Field()
        assert self.parent is not None
        for comp in self.parent.components:
            if comp in ('grid', 'rock', 'tables', 'aquifers'):
                setattr(tmp_model, comp, getattr(self.parent, comp).copy())
        result_dates_restart = self.result_dates
        states_dates_parent = self.parent.states.dates
        states_dates_restart = self.states.dates
        states_tmp = States()
        for attr in self.states.attributes:
            setattr(states_tmp, attr,
                np.concatenate(
                    (getattr(self.parent.states, attr)[states_dates_parent < states_dates_restart[0]],
                     getattr(self.states, attr)))
            )
        tmp_model.states = states_tmp
        wells_tmp = self.parent.wells
        for node in wells_tmp:
            if node.is_main_branch:
                for attr in node.attributes:
                    if attr in ('EVENTS', 'RESULTS', 'PERF', 'COMPDAT', 'WCONPROD', 'WCONINJE'):
                        setattr(node, attr,
                                pd.concat((
                                    getattr(node, attr)[getattr(node, attr)['DATE'] < result_dates_restart[0]],
                                    getattr(self.wells[node.name], attr)
                                ), ignore_index=True))
        tmp_model.wells = wells_tmp
        tmp_model.meta.update(self.parent.meta)
        tmp_model.meta['DATES'] = self.parent.meta['DATES'].append(self.meta['DATES'])
        return tmp_model
