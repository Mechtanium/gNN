"""Rock component."""
import numpy as np

from .base_spatial import SpatialComponent
from .decorators import apply_to_each_input
from .utils import get_single_path
from .parse_utils import read_ecl_bin


class Rock(SpatialComponent):
    """Rock component of geological model."""

    def _load_ecl_binary(self, path_to_results, attrs, basename, logger=None):
        path = get_single_path(path_to_results, basename + '.INIT', logger)
        if path is None:
            return
        sections = read_ecl_bin(path, attrs, logger=logger)

        for k in ['PORO', 'PERMX', 'PERMY', 'PERMZ', "KRW"]:
            if (k in attrs) and (k in sections):
                setattr(self, k, sections[k])
            self.state.binary_attributes.append(k)

    @apply_to_each_input
    def _to_spatial(self, attr):
        """Spatial order 'F' transformations."""
        dimens = self.field.grid.dimens
        self.pad_na(attr=attr)
        return self.reshape(attr=attr, newshape=dimens, order='F', inplace=True)

    def _make_data_dump(self, attr, fmt=None, float_dtype=None, **kwargs):
        """Prepare data for dump."""
        if fmt.upper() != 'HDF5':
            return super()._make_data_dump(attr, fmt=fmt, **kwargs)
        data = self.ravel(attr=attr)
        return data if float_dtype is None else data.astype(float_dtype)

    @apply_to_each_input
    def pad_na(self, attr, fill_na=0., inplace=True):
        """Add dummy cells into the rock vector in the positions of non-active cells if necessary.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be padded with non-active cells.
        actnum: array-like of type bool
            Vector representing a mask of active and non-active cells.
        fill_na: float
            Value to be used as filler.
        inplace: bool
            Modify сomponent inplace.

        Returns
        -------
        output : component if inplace else padded attribute.
        """
        data = getattr(self, attr)
        if np.prod(data.shape) == np.prod(self.field.grid.dimens):
            return self if inplace else data
        actnum = self.field.grid.actnum
        if data.ndim > 1:
            raise ValueError('Data should be ravel for padding.')
        padded_data = np.full(shape=(actnum.size,), fill_value=fill_na, dtype=float)
        padded_data[actnum.ravel(order='F')] = data
        if inplace:
            setattr(self, attr, padded_data)
            return self
        return padded_data

    @apply_to_each_input
    def strip_na(self, attr):
        """Remove non-active cells from the rock vector.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be stripped

        Returns
        -------
        output : stripped attribute.
        """
        data = self.ravel(attr)
        actnum = self.field.grid.actnum
        if data.size == np.sum(actnum):
            return data
        stripped_data = data[actnum.ravel(order='F')]
        return stripped_data

