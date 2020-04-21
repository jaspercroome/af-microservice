import re
import warnings
import operator

import numpy as np
from numpy.lib.stride_tricks import as_strided
import resampy
import audioread
import six

from six import itertools
from warnings import warn

from . import _csparsetools
from . import special

__all__ = ['load','get_duration','cqt','hpss']


# scipy -----------------------------------------------------------------------
INT_TYPES = (int, np.integer)

supported_dtypes = ['bool', 'int8', 'uint8', 'short', 'ushort', 'intc',
                    'uintc', 'longlong', 'ulonglong', 'single', 'double',
                    'longdouble', 'csingle', 'cdouble', 'clongdouble']
supported_dtypes = [np.typeDict[x] for x in supported_dtypes]

_formats = {'csc': [0, "Compressed Sparse Column"],
            'csr': [1, "Compressed Sparse Row"],
            'dok': [2, "Dictionary Of Keys"],
            'lil': [3, "List of Lists"],
            'dod': [4, "Dictionary of Dictionaries"],
            'sss': [5, "Symmetric Sparse Skyline"],
            'coo': [6, "COOrdinate"],
            'lba': [7, "Linpack BAnded"],
            'egd': [8, "Ellpack-itpack Generalized Diagonal"],
            'dia': [9, "DIAgonal"],
            'bsr': [10, "Block Sparse Row"],
            'msr': [11, "Modified compressed Sparse Row"],
            'bsc': [12, "Block Sparse Column"],
            'msc': [13, "Modified compressed Sparse Column"],
            'ssk': [14, "Symmetric SKyline"],
            'nsk': [15, "Nonsymmetric SKyline"],
            'jad': [16, "JAgged Diagonal"],
            'uss': [17, "Unsymmetric Sparse Skyline"],
            'vbr': [18, "Variable Block Row"],
            'und': [19, "Undefined"]
            }

_upcast_memo = {}

MAXPRINT = 50

def isspmatrix(x):
    """Is x of a sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse matrix

    Returns
    -------
    bool
        True if x is a sparse matrix, False otherwise

    Notes
    -----
    issparse and isspmatrix are aliases for the same function.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, isspmatrix
    >>> isspmatrix(csr_matrix([[5]]))
    True

    >>> from scipy.sparse import isspmatrix
    >>> isspmatrix(5)
    False
    """
    return isinstance(x, spmatrix)

def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)

def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            warnings.warn("Inexact indices into sparse matrices are deprecated",
                          DeprecationWarning)
        return loose_int
    return True

def isdense(x):
    return isinstance(x, np.ndarray)

def isshape(x, nonneg=False):
    """Is x a valid 2-tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except Exception:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if np.ndim(M) == 0 and np.ndim(N) == 0:
                if not nonneg or (M >= 0 and N >= 0):
                    return True
        return False

def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples
    --------

    >>> upcast('int32')
    <type 'numpy.int32'>
    >>> upcast('bool')
    <type 'numpy.bool_'>
    >>> upcast('int32','float32')
    <type 'numpy.float64'>
    >>> upcast('bool',complex,float)
    <type 'numpy.complex128'>

    """

    t = _upcast_memo.get(hash(args))
    if t is not None:
        return t

    upcast = np.find_common_type(args, [])

    for t in supported_dtypes:
        if np.can_cast(upcast, t):
            _upcast_memo[hash(args)] = t
            return t

    raise TypeError('no supported conversion for types: %r' % (args,))

def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype

def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    t = upcast(*map(np.dtype, args))
    _upcast_memo[args] = t
    return t

def _unpack_index(index):
    """ Parse index. Always return a tuple of the form (row, col).
    Valid type for row/col is integer, slice, or array of integers.
    """
    # First, check if indexing with single boolean matrix.
    if (isinstance(index, (spmatrix, np.ndarray)) and
            index.ndim == 2 and index.dtype.kind == 'b'):
        return index.nonzero()

    # Parse any ellipses.
    index = _check_ellipsis(index)

    # Next, parse the tuple or object
    if isinstance(index, tuple):
        if len(index) == 2:
            row, col = index
        elif len(index) == 1:
            row, col = index[0], slice(None)
        else:
            raise IndexError('invalid number of indices')
    else:
        row, col = index, slice(None)

    # Next, check for validity and transform the index as needed.
    if isspmatrix(row) or isspmatrix(col):
        # Supporting sparse boolean indexing with both row and col does
        # not work because spmatrix.ndim is always 2.
        raise IndexError(
            'Indexing with sparse matrices is not supported '
            'except boolean indexing where matrix and index '
            'are equal shapes.')
    if isinstance(row, np.ndarray) and row.dtype.kind == 'b':
        row = _boolean_index_to_array(row)
    if isinstance(col, np.ndarray) and col.dtype.kind == 'b':
        col = _boolean_index_to_array(col)
    return row, col

def _check_ellipsis(index):
    """Process indices with Ellipsis. Returns modified index."""
    if index is Ellipsis:
        return (slice(None), slice(None))

    if not isinstance(index, tuple):
        return index

    # TODO: Deprecate this multiple-ellipsis handling,
    #       as numpy no longer supports it.

    # Find first ellipsis.
    for j, v in enumerate(index):
        if v is Ellipsis:
            first_ellipsis = j
            break
    else:
        return index

    # Try to expand it using shortcuts for common cases
    if len(index) == 1:
        return (slice(None), slice(None))
    if len(index) == 2:
        if first_ellipsis == 0:
            if index[1] is Ellipsis:
                return (slice(None), slice(None))
            return (slice(None), index[1])
        return (index[0], slice(None))

    # Expand it using a general-purpose algorithm
    tail = []
    for v in index[first_ellipsis+1:]:
        if v is not Ellipsis:
            tail.append(v)
    nd = first_ellipsis + len(tail)
    nslice = max(0, 2 - nd)
    return index[:first_ellipsis] + (slice(None),)*nslice + tuple(tail)

def _boolean_index_to_array(idx):
    if idx.ndim > 1:
        raise IndexError('invalid index shape')
    return idx.nonzero()[0]

def getdtype(dtype, a=None, default=None):
    """Function used to simplify argument processing.  If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument.  If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    # TODO is this really what we want?
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError:
            if default is not None:
                newdtype = np.dtype(default)
            else:
                raise TypeError("could not interpret data type")
    else:
        newdtype = np.dtype(dtype)
        if newdtype == np.object_:
            warnings.warn("object dtype is not supported by sparse matrices")

    return newdtype

def get_sum_dtype(dtype):
    """Mimic numpy's casting for np.sum"""
    if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
        return np.uint
    if np.can_cast(dtype, np.int_):
        return np.int_
    return dtype

def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """

    int32min = np.iinfo(np.int32).min
    int32max = np.iinfo(np.int32).max

    dtype = np.intc
    if maxval is not None:
        if maxval > int32max:
            dtype = np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if not np.can_cast(arr.dtype, np.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed
                        continue

            dtype = np.int64
            break

    return dtype

def to_native(A):
    return np.asarray(A, dtype=A.dtype.newbyteorder('native'))

def matrix(*args, **kwargs):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            'ignore', '.*the matrix subclass is not the recommended way.*')
        return np.matrix(*args, **kwargs)

def asmatrix(*args, **kwargs):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            'ignore', '.*the matrix subclass is not the recommended way.*')
        return np.asmatrix(*args, **kwargs)

def validateaxis(axis):
    if axis is not None:
        axis_type = type(axis)

        # In NumPy, you can pass in tuples for 'axis', but they are
        # not very useful for sparse matrices given their limited
        # dimensions, so let's make it explicit that they are not
        # allowed to be passed in
        if axis_type == tuple:
            raise TypeError(("Tuples are not accepted for the 'axis' "
                             "parameter. Please pass in one of the "
                             "following: {-2, -1, 0, 1, None}."))

        # If not a tuple, check that the provided axis is actually
        # an integer and raise a TypeError similar to NumPy's
        if not np.issubdtype(np.dtype(axis_type), np.integer):
            raise TypeError("axis must be an integer, not {name}"
                            .format(name=axis_type.__name__))

        if not (-2 <= axis <= 1):
            raise ValueError("axis out of range")

def _prepare_index_for_memoryview(i, j, x=None):
    """
    Convert index and data arrays to form suitable for passing to the
    Cython fancy getset routines.

    The conversions are necessary since to (i) ensure the integer
    index arrays are in one of the accepted types, and (ii) to ensure
    the arrays are writable so that Cython memoryview support doesn't
    choke on them.

    Parameters
    ----------
    i, j
        Index arrays
    x : optional
        Data arrays

    Returns
    -------
    i, j, x
        Re-formatted arrays (x is omitted, if input was None)

    """
    if i.dtype > j.dtype:
        j = j.astype(i.dtype)
    elif i.dtype < j.dtype:
        i = i.astype(j.dtype)

    if not i.flags.writeable or i.dtype not in (np.int32, np.int64):
        i = i.astype(np.intp)
    if not j.flags.writeable or j.dtype not in (np.int32, np.int64):
        j = j.astype(np.intp)

    if x is not None:
        if not x.flags.writeable:
            x = x.copy()
        return i, j, x
    else:
        return i, j

def _prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array

class SparseWarning(Warning):
    pass

class SparseFormatWarning(SparseWarning):
    pass

class SparseEfficiencyWarning(SparseWarning):
    pass

class IndexMixin(object):
    """
    This class provides common dispatching and validation logic for indexing.
    """
    def __getitem__(self, key):
        row, col = self._validate_indices(key)
        # Dispatch to specialized methods.
        if isinstance(row, INT_TYPES):
            if isinstance(col, INT_TYPES):
                return self._get_intXint(row, col)
            elif isinstance(col, slice):
                return self._get_intXslice(row, col)
            elif col.ndim == 1:
                return self._get_intXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif isinstance(row, slice):
            if isinstance(col, INT_TYPES):
                return self._get_sliceXint(row, col)
            elif isinstance(col, slice):
                if row == slice(None) and row == col:
                    return self.copy()
                return self._get_sliceXslice(row, col)
            elif col.ndim == 1:
                return self._get_sliceXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif row.ndim == 1:
            if isinstance(col, INT_TYPES):
                return self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                return self._get_arrayXslice(row, col)
        else:  # row.ndim == 2
            if isinstance(col, INT_TYPES):
                return self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                raise IndexError('index results in >2 dimensions')
            elif row.shape[1] == 1 and col.ndim == 1:
                # special case for outer indexing
                return self._get_columnXarray(row[:,0], col)

        # The only remaining case is inner (fancy) indexing
        row, col = _broadcast_arrays(row, col)
        if row.shape != col.shape:
            raise IndexError('number of row and column indices differ')
        if row.size == 0:
            return self.__class__(np.atleast_2d(row).shape, dtype=self.dtype)
        return self._get_arrayXarray(row, col)

    def __setitem__(self, key, x):
        row, col = self._validate_indices(key)

        if isinstance(row, INT_TYPES) and isinstance(col, INT_TYPES):
            x = np.asarray(x, dtype=self.dtype)
            if x.size != 1:
                raise ValueError('Trying to assign a sequence to an item')
            self._set_intXint(row, col, x.flat[0])
            return

        if isinstance(row, slice):
            row = np.arange(*row.indices(self.shape[0]))[:, None]
        else:
            row = np.atleast_1d(row)

        if isinstance(col, slice):
            col = np.arange(*col.indices(self.shape[1]))[None, :]
            if row.ndim == 1:
                row = row[:, None]
        else:
            col = np.atleast_1d(col)

        i, j = _broadcast_arrays(row, col)
        if i.shape != j.shape:
            raise IndexError('number of row and column indices differ')

        if isspmatrix(x):
            if i.ndim == 1:
                # Inner indexing, so treat them like row vectors.
                i = i[None]
                j = j[None]
            broadcast_row = x.shape[0] == 1 and i.shape[0] != 1
            broadcast_col = x.shape[1] == 1 and i.shape[1] != 1
            if not ((broadcast_row or x.shape[0] == i.shape[0]) and
                    (broadcast_col or x.shape[1] == i.shape[1])):
                raise ValueError('shape mismatch in assignment')
            if x.size == 0:
                return
            x = x.tocoo(copy=True)
            x.sum_duplicates()
            self._set_arrayXarray_sparse(i, j, x)
        else:
            # Make x and i into the same shape
            x = np.asarray(x, dtype=self.dtype)
            x, _ = _broadcast_arrays(x, i)
            if x.size == 0:
                return
            x = x.reshape(i.shape)
            self._set_arrayXarray(i, j, x)

    def _validate_indices(self, key):
        M, N = self.shape
        row, col = _unpack_index(key)

        if isintlike(row):
            row = int(row)
            if row < -M or row >= M:
                raise IndexError('row index (%d) out of range' % row)
            if row < 0:
                row += M
        elif not isinstance(row, slice):
            row = self._asindices(row, M)

        if isintlike(col):
            col = int(col)
            if col < -N or col >= N:
                raise IndexError('column index (%d) out of range' % col)
            if col < 0:
                col += N
        elif not isinstance(col, slice):
            col = self._asindices(col, N)

        return row, col

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.

        Subclasses that need special validation can override this method.
        """
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError):
            raise IndexError('invalid index')

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')

        if x.size == 0:
            return x

        # Check bounds
        max_indx = x.max()
        if max_indx >= length:
            raise IndexError('index (%d) out of range' % max_indx)

        min_indx = x.min()
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError('index (%d) out of range' % min_indx)
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    def getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.
        """
        M, N = self.shape
        i = int(i)
        if i < -M or i >= M:
            raise IndexError('index (%d) out of range' % i)
        if i < 0:
            i += M
        return self._get_intXslice(i, slice(None))

    def getcol(self, i):
        """Return a copy of column i of the matrix, as a (m x 1) column vector.
        """
        M, N = self.shape
        i = int(i)
        if i < -N or i >= N:
            raise IndexError('index (%d) out of range' % i)
        if i < 0:
            i += N
        return self._get_sliceXint(slice(None), i)

    def _get_intXint(self, row, col):
        raise NotImplementedError()

    def _get_intXarray(self, row, col):
        raise NotImplementedError()

    def _get_intXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXint(self, row, col):
        raise NotImplementedError()

    def _get_sliceXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXint(self, row, col):
        raise NotImplementedError()

    def _get_arrayXslice(self, row, col):
        raise NotImplementedError()

    def _get_columnXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXarray(self, row, col):
        raise NotImplementedError()

    def _set_intXint(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray_sparse(self, row, col, x):
        # Fall back to densifying x
        x = np.asarray(x.toarray(), dtype=self.dtype)
        x, _ = _broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)

class _minmax_mixin(object):
    """Mixin for min and max methods.

    These are not implemented for dia_matrix, hence the separate class.
    """

    def _min_or_max_axis(self, axis, min_or_max):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        major_index, value = mat._minor_reduce(min_or_max)
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)

        mask = value != 0
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        from . import coo_matrix
        if axis == 0:
            return coo_matrix((value, (np.zeros(len(value)), major_index)),
                              dtype=self.dtype, shape=(1, M))
        else:
            return coo_matrix((value, (major_index, np.zeros(len(value)))),
                              dtype=self.dtype, shape=(M, 1))

    def _min_or_max(self, axis, out, min_or_max):
        if out is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'out' parameter."))

        validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")

            zero = self.dtype.type(0)
            if self.nnz == 0:
                return zero
            m = min_or_max.reduce(self._deduped_data().ravel())
            if self.nnz != np.prod(self.shape):
                m = min_or_max(zero, m)
            return m

        if axis < 0:
            axis += 2

        if (axis == 0) or (axis == 1):
            return self._min_or_max_axis(axis, min_or_max)
        else:
            raise ValueError("axis out of range")

    def _arg_min_or_max_axis(self, axis, op, compare):
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized "
                             "dimension.")

        if axis < 0:
            axis += 2

        zero = self.dtype.type(0)

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        ret_size, line_size = mat._swap(mat.shape)
        ret = np.zeros(ret_size, dtype=int)

        nz_lines, = np.nonzero(np.diff(mat.indptr))
        for i in nz_lines:
            p, q = mat.indptr[i:i + 2]
            data = mat.data[p:q]
            indices = mat.indices[p:q]
            am = op(data)
            m = data[am]
            if compare(m, zero) or q - p == line_size:
                ret[i] = indices[am]
            else:
                zero_ind = _find_missing_index(indices, line_size)
                if m == zero:
                    ret[i] = min(am, zero_ind)
                else:
                    ret[i] = zero_ind

        if axis == 1:
            ret = ret.reshape(-1, 1)

        return matrix(ret)

    def _arg_min_or_max(self, axis, out, op, compare):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to "
                                 "an empty matrix.")

            if self.nnz == 0:
                return 0
            else:
                zero = self.dtype.type(0)
                mat = self.tocoo()
                mat.sum_duplicates()
                am = op(mat.data)
                m = mat.data[am]

                if compare(m, zero):
                    return mat.row[am] * mat.shape[1] + mat.col[am]
                else:
                    size = np.prod(mat.shape)
                    if size == mat.nnz:
                        return am
                    else:
                        ind = mat.row * mat.shape[1] + mat.col
                        zero_ind = _find_missing_index(ind, size)
                        if m == zero:
                            return min(zero_ind, am)
                        else:
                            return zero_ind

        return self._arg_min_or_max_axis(axis, op, compare)

    def max(self, axis=None, out=None):
        """
        Return the maximum of the matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the maximum over all the matrix elements, returning
            a scalar (i.e. `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        min : The minimum value of a sparse matrix along a given axis.
        numpy.matrix.max : NumPy's implementation of 'max' for matrices

        """
        return self._min_or_max(axis, out, np.maximum)

    def min(self, axis=None, out=None):
        """
        Return the minimum of the matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the minimum over all the matrix elements, returning
            a scalar (i.e. `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        max : The maximum value of a sparse matrix along a given axis.
        numpy.matrix.min : NumPy's implementation of 'min' for matrices

        """
        return self._min_or_max(axis, out, np.minimum)

    def argmax(self, axis=None, out=None):
        """Return indices of maximum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several maximum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmax is computed. If None (default), index
            of the maximum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        ind : numpy.matrix or int
            Indices of maximum elements. If matrix, its size along `axis` is 1.
        """
        return self._arg_min_or_max(axis, out, np.argmax, np.greater)

    def argmin(self, axis=None, out=None):
        """Return indices of minimum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several minimum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmin is computed. If None (default), index
            of the minimum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
         ind : numpy.matrix or int
            Indices of minimum elements. If matrix, its size along `axis` is 1.
        """
        return self._arg_min_or_max(axis, out, np.argmin, np.less)

class spmatrix(object):
    """ This class provides a base class for all sparse matrices.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    """

    __array_priority__ = 10.1
    ndim = 2

    def __init__(self, maxprint=MAXPRINT):
        self._shape = None
        if self.__class__.__name__ == 'spmatrix':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        self.maxprint = maxprint

    def set_shape(self, shape):
        """See `reshape`."""
        # Make sure copy is False since this is in place
        # Make sure format is unchanged because we are doing a __dict__ swap
        new_matrix = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_matrix.__dict__

    def get_shape(self):
        """Get shape of a matrix."""
        return self._shape

    shape = property(fget=get_shape, fset=set_shape)

    def reshape(self, *args, **kwargs):
        """reshape(self, shape, order='C', copy=False)

        Gives a new shape to a sparse matrix without changing its data.

        Parameters
        ----------
        shape : length-2 tuple of ints
            The new shape should be compatible with the original shape.
        order : {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g. read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g. read entire first column, then
            second column, etc.
        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse matrix being used.

        Returns
        -------
        reshaped_matrix : sparse matrix
            A sparse matrix with the given `shape`, not necessarily of the same
            format as the current object.

        See Also
        --------
        numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                               matrices
        """
        # If the shape already matches, don't bother doing an actual reshape
        # Otherwise, the default is to convert to COO and use its reshape
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        return self.tocoo(copy=copy).reshape(shape, order=order, copy=False)

    def resize(self, shape):
        """Resize the matrix in-place to dimensions given by ``shape``

        Any elements that lie within the new shape will remain at the same
        indices, while non-zero elements lying outside the new shape are
        removed.

        Parameters
        ----------
        shape : (int, int)
            number of rows and columns in the new matrix

        Notes
        -----
        The semantics are not identical to `numpy.ndarray.resize` or
        `numpy.resize`.  Here, the same data will be maintained at each index
        before and after reshape, if that index is within the new bounds.  In
        numpy, resizing maintains contiguity of the array, moving elements
        around in the logical matrix but not within a flattened representation.

        We give no guarantees about whether the underlying data attributes
        (arrays, etc.) will be modified in place or replaced with new objects.
        """
        # As an inplace operation, this requires implementation in each format.
        raise NotImplementedError(
            '{}.resize is not implemented'.format(type(self).__name__))

    def astype(self, dtype, casting='unsafe', copy=True):
        """Cast the matrix elements to a specified type.

        Parameters
        ----------
        dtype : string or numpy dtype
            Typecode or data-type to which to cast the data.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.
            Defaults to 'unsafe' for backwards compatibility.
            'no' means the data types should not be cast at all.
            'equiv' means only byte-order changes are allowed.
            'safe' means only casts which can preserve values are allowed.
            'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
            'unsafe' means any data conversions may be done.
        copy : bool, optional
            If `copy` is `False`, the result might share some memory with this
            matrix. If `copy` is `True`, it is guaranteed that the result and
            this matrix do not share any memory.
        """

        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            return self.tocsr().astype(
                dtype, casting=casting, copy=copy).asformat(self.format)
        elif copy:
            return self.copy()
        else:
            return self

    def asfptype(self):
        """Upcast matrix to a floating point format (if necessary)"""

        fp_types = ['f', 'd', 'F', 'D']

        if self.dtype.char in fp_types:
            return self
        else:
            for fp_type in fp_types:
                if self.dtype <= np.dtype(fp_type):
                    return self.astype(fp_type)

            raise TypeError('cannot upcast [%s] to a floating '
                            'point format' % self.dtype.name)

    def __iter__(self):
        for r in six.xrange(self.shape[0]):
            yield self[r, :]

    def getmaxprint(self):
        """Maximum number of elements to display when printed."""
        return self.maxprint

    def count_nonzero(self):
        """Number of non-zero entries, equivalent to

        np.count_nonzero(a.toarray())

        Unlike getnnz() and the nnz property, which return the number of stored
        entries (the length of the data attribute), this method counts the
        actual number of non-zero entries in data.
        """
        raise NotImplementedError("count_nonzero not implemented for %s." %
                                  self.__class__.__name__)

    def getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole matrix, in
            each column, or in each row.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        raise NotImplementedError("getnnz not implemented for %s." %
                                  self.__class__.__name__)

    @property
    def nnz(self):
        """Number of stored values, including explicit zeros.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        return self.getnnz()

    def getformat(self):
        """Format of a matrix representation as a string."""
        return getattr(self, 'format', 'und')

    def __repr__(self):
        _, format_name = _formats[self.getformat()]
        return "<%dx%d sparse matrix of type '%s'\n" \
               "\twith %d stored elements in %s format>" % \
               (self.shape + (self.dtype.type, self.nnz, format_name))

    def __str__(self):
        maxprint = self.getmaxprint()

        A = self.tocoo()

        # helper function, outputs "(i,j)  v"
        def tostr(row, col, data):
            triples = zip(list(zip(row, col)), data)
            return '\n'.join([('  %s\t%s' % t) for t in triples])

        if self.nnz > maxprint:
            half = maxprint // 2
            out = tostr(A.row[:half], A.col[:half], A.data[:half])
            out += "\n  :\t:\n"
            half = maxprint - maxprint//2
            out += tostr(A.row[-half:], A.col[-half:], A.data[-half:])
        else:
            out = tostr(A.row, A.col, A.data)

        return out

    def __bool__(self):  # Simple -- other ideas?
        if self.shape == (1, 1):
            return self.nnz != 0
        else:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all().")
    __nonzero__ = __bool__

    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  But for some uses the number of
    # non-zeros is more important.  For now, raise an exception!
    def __len__(self):
        raise TypeError("sparse matrix length is ambiguous; use getnnz()"
                        " or shape[0]")

    def asformat(self, format, copy=False):
        """Return this matrix in the passed format.

        Parameters
        ----------
        format : {str, None}
            The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : This matrix in the passed format.
        """
        if format is None or format == self.format:
            if copy:
                return self.copy()
            else:
                return self
        else:
            try:
                convert_method = getattr(self, 'to' + format)
            except AttributeError:
                raise ValueError('Format {} is unknown.'.format(format))

            # Forward the copy kwarg, if it's accepted.
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    ###################################################################
    #  NOTE: All arithmetic operations use csr_matrix by default.
    # Therefore a new sparse matrix format just needs to define a
    # .tocsr() method to provide arithmetic support.  Any of these
    # methods can be overridden for efficiency.
    ####################################################################

    def multiply(self, other):
        """Point-wise multiplication by another matrix
        """
        return self.tocsr().multiply(other)

    def maximum(self, other):
        """Element-wise maximum between this and another matrix."""
        return self.tocsr().maximum(other)

    def minimum(self, other):
        """Element-wise minimum between this and another matrix."""
        return self.tocsr().minimum(other)

    def dot(self, other):
        """Ordinary dot product

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1], dtype=int64)

        """
        return self * other

    def power(self, n, dtype=None):
        """Element-wise power."""
        return self.tocsr().power(n, dtype=dtype)

    def __eq__(self, other):
        return self.tocsr().__eq__(other)

    def __ne__(self, other):
        return self.tocsr().__ne__(other)

    def __lt__(self, other):
        return self.tocsr().__lt__(other)

    def __gt__(self, other):
        return self.tocsr().__gt__(other)

    def __le__(self, other):
        return self.tocsr().__le__(other)

    def __ge__(self, other):
        return self.tocsr().__ge__(other)

    def __abs__(self):
        return abs(self.tocsr())

    def __round__(self, ndigits=0):
        return round(self.tocsr(), ndigits=ndigits)

    def _add_sparse(self, other):
        return self.tocsr()._add_sparse(other)

    def _add_dense(self, other):
        return self.tocoo()._add_dense(other)

    def _sub_sparse(self, other):
        return self.tocsr()._sub_sparse(other)

    def _sub_dense(self, other):
        return self.todense() - other

    def _rsub_dense(self, other):
        # note: this can't be replaced by other + (-self) for unsigned types
        return other - self.todense()

    def __add__(self, other):  # self + other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            # Now we would add this scalar to every element.
            raise NotImplementedError('adding a nonzero scalar to a '
                                      'sparse matrix is not supported')
        elif isspmatrix(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._add_sparse(other)
        elif isdense(other):
            other = broadcast_to(other, self.shape)
            return self._add_dense(other)
        else:
            return NotImplemented

    def __radd__(self,other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):  # self - other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError('subtracting a nonzero scalar from a '
                                      'sparse matrix is not supported')
        elif isspmatrix(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._sub_sparse(other)
        elif isdense(other):
            other = broadcast_to(other, self.shape)
            return self._sub_dense(other)
        else:
            return NotImplemented

    def __rsub__(self,other):  # other - self
        if isscalarlike(other):
            if other == 0:
                return -self.copy()
            raise NotImplementedError('subtracting a sparse matrix from a '
                                      'nonzero scalar is not supported')
        elif isdense(other):
            other = broadcast_to(other, self.shape)
            return self._rsub_dense(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """interpret other and call one of the following

        self._mul_scalar()
        self._mul_vector()
        self._mul_multivector()
        self._mul_sparse_matrix()
        """

        M, N = self.shape

        if other.__class__ is np.ndarray:
            # Fast path for the most common case
            if other.shape == (N,):
                return self._mul_vector(other)
            elif other.shape == (N, 1):
                return self._mul_vector(other.ravel()).reshape(M, 1)
            elif other.ndim == 2 and other.shape[0] == N:
                return self._mul_multivector(other)

        if isscalarlike(other):
            # scalar value
            return self._mul_scalar(other)

        if issparse(other):
            if self.shape[1] != other.shape[0]:
                raise ValueError('dimension mismatch')
            return self._mul_sparse_matrix(other)

        # If it's a list or whatever, treat it like a matrix
        other_a = np.asanyarray(other)

        if other_a.ndim == 0 and other_a.dtype == np.object_:
            # Not interpretable as an array; return NotImplemented so that
            # other's __rmul__ can kick in if that's implemented.
            return NotImplemented

        try:
            other.shape
        except AttributeError:
            other = other_a

        if other.ndim == 1 or other.ndim == 2 and other.shape[1] == 1:
            # dense row or column vector
            if other.shape != (N,) and other.shape != (N, 1):
                raise ValueError('dimension mismatch')

            result = self._mul_vector(np.ravel(other))

            if isinstance(other, np.matrix):
                result = asmatrix(result)

            if other.ndim == 2 and other.shape[1] == 1:
                # If 'other' was an (nx1) column vector, reshape the result
                result = result.reshape(-1, 1)

            return result

        elif other.ndim == 2:
            ##
            # dense 2D array or matrix ("multivector")

            if other.shape[0] != self.shape[1]:
                raise ValueError('dimension mismatch')

            result = self._mul_multivector(np.asarray(other))

            if isinstance(other, np.matrix):
                result = asmatrix(result)

            return result

        else:
            raise ValueError('could not interpret dimensions')

    # by default, use CSR for __mul__ handlers
    def _mul_scalar(self, other):
        return self.tocsr()._mul_scalar(other)

    def _mul_vector(self, other):
        return self.tocsr()._mul_vector(other)

    def _mul_multivector(self, other):
        return self.tocsr()._mul_multivector(other)

    def _mul_sparse_matrix(self, other):
        return self.tocsr()._mul_sparse_matrix(other)

    def __rmul__(self, other):  # other * self
        if isscalarlike(other):
            return self.__mul__(other)
        else:
            # Don't use asarray unless we have to
            try:
                tr = other.transpose()
            except AttributeError:
                tr = np.asarray(other).transpose()
            return (self.transpose() * tr).transpose()

    #####################################
    # matmul (@) operator (Python 3.5+) #
    #####################################

    def __matmul__(self, other):
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    ####################
    # Other Arithmetic #
    ####################

    def _divide(self, other, true_divide=False, rdivide=False):
        if isscalarlike(other):
            if rdivide:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())

            if true_divide and np.can_cast(self.dtype, np.float_):
                return self.astype(np.float_)._mul_scalar(1./other)
            else:
                r = self._mul_scalar(1./other)

                scalar_dtype = np.asarray(other).dtype
                if (np.issubdtype(self.dtype, np.integer) and
                        np.issubdtype(scalar_dtype, np.integer)):
                    return r.astype(self.dtype)
                else:
                    return r

        elif isdense(other):
            if not rdivide:
                if true_divide:
                    return np.true_divide(self.todense(), other)
                else:
                    return np.divide(self.todense(), other)
            else:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())
        elif isspmatrix(other):
            if rdivide:
                return other._divide(self, true_divide, rdivide=False)

            self_csr = self.tocsr()
            if true_divide and np.can_cast(self.dtype, np.float_):
                return self_csr.astype(np.float_)._divide_sparse(other)
            else:
                return self_csr._divide_sparse(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self._divide(other, true_divide=True)

    def __div__(self, other):
        # Always do true division
        return self._divide(other, true_divide=True)

    def __rtruediv__(self, other):
        # Implementing this as the inverse would be too magical -- bail out
        return NotImplemented

    def __rdiv__(self, other):
        # Implementing this as the inverse would be too magical -- bail out
        return NotImplemented

    def __neg__(self):
        return -self.tocsr()

    def __iadd__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        return NotImplemented

    def __pow__(self, other):
        if self.shape[0] != self.shape[1]:
            raise TypeError('matrix is not square')

        if isintlike(other):
            other = int(other)
            if other < 0:
                raise ValueError('exponent must be >= 0')

            if other == 0:
                from .construct import eye
                return eye(self.shape[0], dtype=self.dtype)
            elif other == 1:
                return self.copy()
            else:
                tmp = self.__pow__(other//2)
                if (other % 2):
                    return self * tmp * tmp
                else:
                    return tmp * tmp
        elif isscalarlike(other):
            raise ValueError('exponent must be an integer')
        else:
            return NotImplemented

    def __getattr__(self, attr):
        if attr == 'A':
            return self.toarray()
        elif attr == 'T':
            return self.transpose()
        elif attr == 'H':
            return self.getH()
        elif attr == 'real':
            return self._real()
        elif attr == 'imag':
            return self._imag()
        elif attr == 'size':
            return self.getnnz()
        else:
            raise AttributeError(attr + " not found")

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse matrix.

        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse matrix
            being used.

        Returns
        -------
        p : `self` with the dimensions reversed.

        See Also
        --------
        numpy.matrix.transpose : NumPy's implementation of 'transpose'
                                 for matrices
        """
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False)

    def conj(self, copy=True):
        """Element-wise complex conjugation.

        If the matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        """
        if np.issubdtype(self.dtype, np.complexfloating):
            return self.tocsr(copy=copy).conj(copy=False)
        elif copy:
            return self.copy()
        else:
            return self

    def conjugate(self, copy=True):
        return self.conj(copy=copy)

    conjugate.__doc__ = conj.__doc__

    # Renamed conjtranspose() -> getH() for compatibility with dense matrices
    def getH(self):
        """Return the Hermitian transpose of this matrix.

        See Also
        --------
        numpy.matrix.getH : NumPy's implementation of `getH` for matrices
        """
        return self.transpose().conj()

    def _real(self):
        return self.tocsr()._real()

    def _imag(self):
        return self.tocsr()._imag()

    def nonzero(self):
        """nonzero indices

        Returns a tuple of arrays (row,col) containing the indices
        of the non-zero elements of the matrix.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
        >>> A.nonzero()
        (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))

        """

        # convert to COOrdinate format
        A = self.tocoo()
        nz_mask = A.data != 0
        return (A.row[nz_mask], A.col[nz_mask])

    def getcol(self, j):
        """Returns a copy of column j of the matrix, as an (m x 1) sparse
        matrix (column vector).
        """
        # Spmatrix subclasses should override this method for efficiency.
        # Post-multiply by a (n x 1) column vector 'a' containing all zeros
        # except for a_j = 1
        from .csc import csc_matrix
        n = self.shape[1]
        if j < 0:
            j += n
        if j < 0 or j >= n:
            raise IndexError("index out of bounds")
        col_selector = csc_matrix(([1], [[j], [0]]),
                                  shape=(n, 1), dtype=self.dtype)
        return self * col_selector

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n) sparse
        matrix (row vector).
        """
        # Spmatrix subclasses should override this method for efficiency.
        # Pre-multiply by a (1 x m) row vector 'a' containing all zeros
        # except for a_i = 1
        from .csr import csr_matrix
        m = self.shape[0]
        if i < 0:
            i += m
        if i < 0 or i >= m:
            raise IndexError("index out of bounds")
        row_selector = csr_matrix(([1], [[0], [i]]),
                                  shape=(1, m), dtype=self.dtype)
        return row_selector * self

    # def __array__(self):
    #    return self.toarray()

    def todense(self, order=None, out=None):
        """
        Return a dense matrix representation of this matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', indicating the NumPy default of C-ordered.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-dimensional, optional
            If specified, uses this array (or `numpy.matrix`) as the
            output buffer instead of allocating a new array to
            return. The provided array must have the same shape and
            dtype as the sparse matrix on which you are calling the
            method.

        Returns
        -------
        arr : numpy.matrix, 2-dimensional
            A NumPy matrix object with the same shape and containing
            the same data represented by the sparse matrix, with the
            requested memory order. If `out` was passed and was an
            array (rather than a `numpy.matrix`), it will be filled
            with the appropriate values and returned wrapped in a
            `numpy.matrix` object that shares the same memory.
        """
        return asmatrix(self.toarray(order=order, out=out))

    def toarray(self, order=None, out=None):
        """
        Return a dense ndarray representation of this matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', indicating the NumPy default of C-ordered.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-dimensional, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The provided
            array must have the same shape and dtype as the sparse
            matrix on which you are calling the method. For most
            sparse types, `out` is required to be memory contiguous
            (either C or Fortran ordered).

        Returns
        -------
        arr : ndarray, 2-dimensional
            An array with the same shape and containing the same
            data represented by the sparse matrix, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        """
        return self.tocoo(copy=False).toarray(order=order, out=out)

    # Any sparse matrix format deriving from spmatrix must define one of
    # tocsr or tocoo. The other conversion methods may be implemented for
    # efficiency, but are not required.
    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant csr_matrix.
        """
        return self.tocoo(copy=copy).tocsr(copy=False)

    def todok(self, copy=False):
        """Convert this matrix to Dictionary Of Keys format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant dok_matrix.
        """
        return self.tocoo(copy=copy).todok(copy=False)

    def tocoo(self, copy=False):
        """Convert this matrix to COOrdinate format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant coo_matrix.
        """
        return self.tocsr(copy=False).tocoo(copy=copy)

    def tolil(self, copy=False):
        """Convert this matrix to List of Lists format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant lil_matrix.
        """
        return self.tocsr(copy=False).tolil(copy=copy)

    def todia(self, copy=False):
        """Convert this matrix to sparse DIAgonal format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant dia_matrix.
        """
        return self.tocoo(copy=copy).todia(copy=False)

    def tobsr(self, blocksize=None, copy=False):
        """Convert this matrix to Block Sparse Row format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant bsr_matrix.

        When blocksize=(R, C) is provided, it will be used for construction of
        the bsr_matrix.
        """
        return self.tocsr(copy=False).tobsr(blocksize=blocksize, copy=copy)

    def tocsc(self, copy=False):
        """Convert this matrix to Compressed Sparse Column format.

        With copy=False, the data/indices may be shared between this matrix and
        the resultant csc_matrix.
        """
        return self.tocsr(copy=copy).tocsc(copy=False)

    def copy(self):
        """Returns a copy of this matrix.

        No data/indices will be shared between the returned value and current
        matrix.
        """
        return self.__class__(self, copy=True)

    def sum(self, axis=None, dtype=None, out=None):
        """
        Sum the matrix elements over a given axis.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the matrix elements, returning a scalar
            (i.e. `axis` = `None`).
        dtype : dtype, optional
            The type of the returned matrix and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.

        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices

        """
        validateaxis(axis)

        # We use multiplication by a matrix of ones to achieve this.
        # For some sparse matrix formats more efficient methods are
        # possible -- these should override this function.
        m, n = self.shape

        # Mimic numpy's casting.
        res_dtype = get_sum_dtype(self.dtype)

        if axis is None:
            # sum over rows and columns
            return (self * asmatrix(np.ones(
                (n, 1), dtype=res_dtype))).sum(
                dtype=dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            # sum over columns
            ret = asmatrix(np.ones(
                (1, m), dtype=res_dtype)) * self
        else:
            # sum over rows
            ret = self * asmatrix(
                np.ones((n, 1), dtype=res_dtype))

        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=(), dtype=dtype, out=out)

    def mean(self, axis=None, dtype=None, out=None):
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the matrix elements. The average is taken
        over all elements in the matrix by default, otherwise over the
        specified axis. `float64` intermediate and return values are used
        for integer inputs.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the mean is computed. The default is to compute
            the mean of all elements in the matrix (i.e. `axis` = `None`).
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        m : np.matrix

        See Also
        --------
        numpy.matrix.mean : NumPy's implementation of 'mean' for matrices

        """
        def _is_integral(dtype):
            return (np.issubdtype(dtype, np.integer) or
                    np.issubdtype(dtype, np.bool_))

        validateaxis(axis)

        res_dtype = self.dtype.type
        integral = _is_integral(self.dtype)

        # output dtype
        if dtype is None:
            if integral:
                res_dtype = np.float64
        else:
            res_dtype = np.dtype(dtype).type

        # intermediate dtype for summation
        inter_dtype = np.float64 if integral else res_dtype
        inter_self = self.astype(inter_dtype)

        if axis is None:
            return (inter_self / np.array(
                self.shape[0] * self.shape[1]))\
                .sum(dtype=res_dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            return (inter_self * (1.0 / self.shape[0])).sum(
                axis=0, dtype=res_dtype, out=out)
        else:
            return (inter_self * (1.0 / self.shape[1])).sum(
                axis=1, dtype=res_dtype, out=out)

    def diagonal(self, k=0):
        """Returns the k-th diagonal of the matrix.

        Parameters
        ----------
        k : int, optional
            Which diagonal to get, corresponding to elements a[i, i+k].
            Default: 0 (the main diagonal).

            .. versionadded:: 1.0

        See also
        --------
        numpy.diagonal : Equivalent numpy function.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A.diagonal()
        array([1, 0, 5])
        >>> A.diagonal(k=1)
        array([2, 3])
        """
        return self.tocsr().diagonal(k=k)

    def setdiag(self, values, k=0):
        """
        Set diagonal or off-diagonal elements of the array.

        Parameters
        ----------
        values : array_like
            New values of the diagonal elements.

            Values may have any length.  If the diagonal is longer than values,
            then the remaining diagonal entries will not be set.  If values if
            longer than the diagonal, then the remaining values are ignored.

            If a scalar value is given, all of the diagonal is set to it.

        k : int, optional
            Which off-diagonal to set, corresponding to elements a[i,i+k].
            Default: 0 (the main diagonal).

        """
        M, N = self.shape
        if (k > 0 and k >= N) or (k < 0 and -k >= M):
            raise ValueError("k exceeds matrix dimensions")
        self._setdiag(np.asarray(values), k)

    def _setdiag(self, values, k):
        M, N = self.shape
        if k < 0:
            if values.ndim == 0:
                # broadcast
                max_index = min(M+k, N)
                for i in xrange(max_index):
                    self[i - k, i] = values
            else:
                max_index = min(M+k, N, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i - k, i] = v
        else:
            if values.ndim == 0:
                # broadcast
                max_index = min(M, N-k)
                for i in xrange(max_index):
                    self[i, i + k] = values
            else:
                max_index = min(M, N-k, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i, i + k] = v

    def _process_toarray_args(self, order, out):
        if out is not None:
            if order is not None:
                raise ValueError('order cannot be specified if out '
                                 'is not None')
            if out.shape != self.shape or out.dtype != self.dtype:
                raise ValueError('out array must be same dtype and shape as '
                                 'sparse matrix')
            out[...] = 0.
            return out
        else:
            return np.zeros(self.shape, dtype=self.dtype, order=order)

class _data_matrix(spmatrix):
    def __init__(self):
        spmatrix.__init__(self)

    def _get_dtype(self):
        return self.data.dtype

    def _set_dtype(self, newtype):
        self.data.dtype = newtype
    dtype = property(fget=_get_dtype, fset=_set_dtype)

    def _deduped_data(self):
        if hasattr(self, 'sum_duplicates'):
            self.sum_duplicates()
        return self.data

    def __abs__(self):
        return self._with_data(abs(self._deduped_data()))

    def __round__(self, ndigits=0):
        return self._with_data(np.around(self._deduped_data(), decimals=ndigits))

    def _real(self):
        return self._with_data(self.data.real)

    def _imag(self):
        return self._with_data(self.data.imag)

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('negating a sparse boolean '
                                      'matrix is not supported')
        return self._with_data(-self.data)

    def __imul__(self, other):  # self *= other
        if isscalarlike(other):
            self.data *= other
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):  # self /= other
        if isscalarlike(other):
            recip = 1.0 / other
            self.data *= recip
            return self
        else:
            return NotImplemented

    def astype(self, dtype, casting='unsafe', copy=True):
        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            return self._with_data(
                self._deduped_data().astype(dtype, casting=casting, copy=copy),
                copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    astype.__doc__ = spmatrix.astype.__doc__

    def conj(self, copy=True):
        if np.issubdtype(self.dtype, np.complexfloating):
            return self._with_data(self.data.conj(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    conj.__doc__ = spmatrix.conj.__doc__

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    copy.__doc__ = spmatrix.copy.__doc__

    def count_nonzero(self):
        return np.count_nonzero(self._deduped_data())

    count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__

    def power(self, n, dtype=None):
        """
        This function performs element-wise power.

        Parameters
        ----------
        n : n is a scalar

        dtype : If dtype is not specified, the current dtype will be preserved.
        """
        if not isscalarlike(n):
            raise NotImplementedError("input is not scalar")

        data = self._deduped_data()
        if dtype is not None:
            data = data.astype(dtype)
        return self._with_data(data ** n)

    ###########################
    # Multiplication handlers #
    ###########################

    def _mul_scalar(self, other):
        return self._with_data(self.data * other)

class _cs_matrix(_data_matrix, _minmax_mixin, IndexMixin):
    """base matrix class for compressed row and column oriented matrices"""

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        if isspmatrix(arg1):
            if arg1.format == self.format and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.asformat(self.format)
            self._set_self(arg1)

        elif isinstance(arg1, tuple):
            if isshape(arg1):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self._shape = check_shape(arg1)
                M, N = self.shape
                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                idx_dtype = get_index_dtype(maxval=max(M, N))
                self.data = np.zeros(0, getdtype(dtype, default=float))
                self.indices = np.zeros(0, idx_dtype)
                self.indptr = np.zeros(self._swap((M, N))[0] + 1,
                                       dtype=idx_dtype)
            else:
                if len(arg1) == 2:
                    # (data, ij) format
                    other = self.__class__(coo_matrix(arg1, shape=shape))
                    self._set_self(other)
                elif len(arg1) == 3:
                    # (data, indices, indptr) format
                    (data, indices, indptr) = arg1

                    # Select index dtype large enough to pass array and
                    # scalar parameters to sparsetools
                    maxval = None
                    if shape is not None:
                        maxval = max(shape)
                    idx_dtype = get_index_dtype((indices, indptr),
                                                maxval=maxval,
                                                check_contents=True)

                    self.indices = np.array(indices, copy=copy,
                                            dtype=idx_dtype)
                    self.indptr = np.array(indptr, copy=copy, dtype=idx_dtype)
                    self.data = np.array(data, copy=copy, dtype=dtype)
                else:
                    raise ValueError("unrecognized {}_matrix "
                                     "constructor usage".format(self.format))

        else:
            # must be dense
            try:
                arg1 = np.asarray(arg1)
            except Exception:
                raise ValueError("unrecognized {}_matrix constructor usage"
                                 "".format(self.format))
            self._set_self(self.__class__(coo_matrix(arg1, dtype=dtype)))

        # Read matrix dimensions given, if any
        if shape is not None:
            self._shape = check_shape(shape)
        else:
            if self.shape is None:
                # shape not already set, try to infer dimensions
                try:
                    major_dim = len(self.indptr) - 1
                    minor_dim = self.indices.max() + 1
                except Exception:
                    raise ValueError('unable to infer matrix dimensions')
                else:
                    self._shape = check_shape(self._swap((major_dim,
                                                          minor_dim)))

        if dtype is not None:
            self.data = np.asarray(self.data, dtype=dtype)

        self.check_format(full_check=False)

    def getnnz(self, axis=None):
        if axis is None:
            return int(self.indptr[-1])
        else:
            if axis < 0:
                axis += 2
            axis, _ = self._swap((axis, 1 - axis))
            _, N = self._swap(self.shape)
            if axis == 0:
                return np.bincount(downcast_intp_index(self.indices),
                                   minlength=N)
            elif axis == 1:
                return np.diff(self.indptr)
            raise ValueError('axis out of bounds')

    getnnz.__doc__ = spmatrix.getnnz.__doc__

    def _set_self(self, other, copy=False):
        """take the member variables of other and assign them to self"""

        if copy:
            other = other.copy()

        self.data = other.data
        self.indices = other.indices
        self.indptr = other.indptr
        self._shape = check_shape(other.shape)

    def check_format(self, full_check=True):
        """check whether the matrix format is valid

        Parameters
        ----------
        full_check : bool, optional
            If `True`, rigorous check, O(N) operations. Otherwise
            basic check, O(1) operations (default True).
        """
        # use _swap to determine proper bounds
        major_name = 'column'
        minor_name = 'row'
        # , minor_name = self._swap(('row', 'column'))
        major_dim = self.shape[0]
        minor_dim = self.shape[1]
        # , minor_dim = self._swap(self.shape)

        # index arrays should have integer data types
        if self.indptr.dtype.kind != 'i':
            warn("indptr array has non-integer dtype ({})"
                 "".format(self.indptr.dtype.name), stacklevel=3)
        if self.indices.dtype.kind != 'i':
            warn("indices array has non-integer dtype ({})"
                 "".format(self.indices.dtype.name), stacklevel=3)

        idx_dtype = get_index_dtype((self.indptr, self.indices))
        self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
        self.indices = np.asarray(self.indices, dtype=idx_dtype)
        self.data = to_native(self.data)

        # check array shapes
        for x in [self.data.ndim, self.indices.ndim, self.indptr.ndim]:
            if x != 1:
                raise ValueError('data, indices, and indptr should be 1-D')

        # check index pointer
        if (len(self.indptr) != major_dim + 1):
            raise ValueError("index pointer size ({}) should be ({})"
                             "".format(len(self.indptr), major_dim + 1))
        if (self.indptr[0] != 0):
            raise ValueError("index pointer should start with 0")

        # check index and data arrays
        if (len(self.indices) != len(self.data)):
            raise ValueError("indices and data should have the same size")
        if (self.indptr[-1] > len(self.indices)):
            raise ValueError("Last value of index pointer should be less than "
                             "the size of index and data arrays")

        self.prune()

        if full_check:
            # check format validity (more expensive)
            if self.nnz > 0:
                if self.indices.max() >= minor_dim:
                    raise ValueError("{} index values must be < {}"
                                     "".format(minor_name, minor_dim))
                if self.indices.min() < 0:
                    raise ValueError("{} index values must be >= 0"
                                     "".format(minor_name))
                if np.diff(self.indptr).min() < 0:
                    raise ValueError("index pointer values must form a "
                                     "non-decreasing sequence")

        # if not self.has_sorted_indices():
        #    warn('Indices were not in sorted order.  Sorting indices.')
        #    self.sort_indices()
        #    assert(self.has_sorted_indices())
        # TODO check for duplicates?

    #######################
    # Boolean comparisons #
    #######################

    def _scalar_binopt(self, other, op):
        """Scalar version of self._binopt, for cases in which no new nonzeros
        are added. Produces a new spmatrix in canonical form.
        """
        self.sum_duplicates()
        res = self._with_data(op(self.data, other), copy=True)
        res.eliminate_zeros()
        return res

    def __eq__(self, other):
        # Scalar other.
        if isscalarlike(other):
            if np.isnan(other):
                return self.__class__(self.shape, dtype=np.bool_)

            if other == 0:
                warn("Comparing a sparse matrix with 0 using == is inefficient"
                     ", try using != instead.", SparseEfficiencyWarning,
                     stacklevel=3)
                all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
                inv = self._scalar_binopt(other, operator.ne)
                return all_true - inv
            else:
                return self._scalar_binopt(other, operator.eq)
        # Dense other.
        elif isdense(other):
            return self.todense() == other
        # Sparse other.
        elif isspmatrix(other):
            warn("Comparing sparse matrices using == is inefficient, try using"
                 " != instead.", SparseEfficiencyWarning, stacklevel=3)
            # TODO sparse broadcasting
            if self.shape != other.shape:
                return False
            elif self.format != other.format:
                other = other.asformat(self.format)
            res = self._binopt(other, '_ne_')
            all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
            return all_true - res
        else:
            return False

    def __ne__(self, other):
        # Scalar other.
        if isscalarlike(other):
            if np.isnan(other):
                warn("Comparing a sparse matrix with nan using != is"
                     " inefficient", SparseEfficiencyWarning, stacklevel=3)
                all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
                return all_true
            elif other != 0:
                warn("Comparing a sparse matrix with a nonzero scalar using !="
                     " is inefficient, try using == instead.",
                     SparseEfficiencyWarning, stacklevel=3)
                all_true = self.__class__(np.ones(self.shape), dtype=np.bool_)
                inv = self._scalar_binopt(other, operator.eq)
                return all_true - inv
            else:
                return self._scalar_binopt(other, operator.ne)
        # Dense other.
        elif isdense(other):
            return self.todense() != other
        # Sparse other.
        elif isspmatrix(other):
            # TODO sparse broadcasting
            if self.shape != other.shape:
                return True
            elif self.format != other.format:
                other = other.asformat(self.format)
            return self._binopt(other, '_ne_')
        else:
            return True

    def _inequality(self, other, op, op_name, bad_scalar_msg):
        # Scalar other.
        if isscalarlike(other):
            if 0 == other and op_name in ('_le_', '_ge_'):
                raise NotImplementedError(" >= and <= don't work with 0.")
            elif op(0, other):
                warn(bad_scalar_msg, SparseEfficiencyWarning)
                other_arr = np.empty(self.shape, dtype=np.result_type(other))
                other_arr.fill(other)
                other_arr = self.__class__(other_arr)
                return self._binopt(other_arr, op_name)
            else:
                return self._scalar_binopt(other, op)
        # Dense other.
        elif isdense(other):
            return op(self.todense(), other)
        # Sparse other.
        elif isspmatrix(other):
            # TODO sparse broadcasting
            if self.shape != other.shape:
                raise ValueError("inconsistent shapes")
            elif self.format != other.format:
                other = other.asformat(self.format)
            if op_name not in ('_ge_', '_le_'):
                return self._binopt(other, op_name)

            warn("Comparing sparse matrices using >= and <= is inefficient, "
                 "using <, >, or !=, instead.", SparseEfficiencyWarning)
            all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
            res = self._binopt(other, '_gt_' if op_name == '_le_' else '_lt_')
            return all_true - res
        else:
            raise ValueError("Operands could not be compared.")

    def __lt__(self, other):
        return self._inequality(other, operator.lt, '_lt_',
                                "Comparing a sparse matrix with a scalar "
                                "greater than zero using < is inefficient, "
                                "try using >= instead.")

    def __gt__(self, other):
        return self._inequality(other, operator.gt, '_gt_',
                                "Comparing a sparse matrix with a scalar "
                                "less than zero using > is inefficient, "
                                "try using <= instead.")

    def __le__(self, other):
        return self._inequality(other, operator.le, '_le_',
                                "Comparing a sparse matrix with a scalar "
                                "greater than zero using <= is inefficient, "
                                "try using > instead.")

    def __ge__(self, other):
        return self._inequality(other, operator.ge, '_ge_',
                                "Comparing a sparse matrix with a scalar "
                                "less than zero using >= is inefficient, "
                                "try using < instead.")

    #################################
    # Arithmetic operator overrides #
    #################################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError('Incompatible shapes.')
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        order = self._swap('CF')[0]
        result = np.array(other, dtype=dtype, order=order, copy=True)
        M, N = self._swap(self.shape)
        y = result if result.flags.c_contiguous else result.T
        csr_todense(M, N, self.indptr, self.indices, self.data, y)
        return matrix(result, copy=False)

    def _add_sparse(self, other):
        return self._binopt(other, '_plus_')

    def _sub_sparse(self, other):
        return self._binopt(other, '_minus_')

    def multiply(self, other):
        """Point-wise multiplication by another matrix, vector, or
        scalar.
        """
        # Scalar multiplication.
        if isscalarlike(other):
            return self._mul_scalar(other)
        # Sparse matrix or vector.
        if isspmatrix(other):
            if self.shape == other.shape:
                other = self.__class__(other)
                return self._binopt(other, '_elmul_')
            # Single element.
            elif other.shape == (1, 1):
                return self._mul_scalar(other.toarray()[0, 0])
            elif self.shape == (1, 1):
                return other._mul_scalar(self.toarray()[0, 0])
            # A row times a column.
            elif self.shape[1] == 1 and other.shape[0] == 1:
                return self._mul_sparse_matrix(other.tocsc())
            elif self.shape[0] == 1 and other.shape[1] == 1:
                return other._mul_sparse_matrix(self.tocsc())
            # Row vector times matrix. other is a row.
            elif other.shape[0] == 1 and self.shape[1] == other.shape[1]:
                other = dia_matrix((other.toarray().ravel(), [0]),
                                   shape=(other.shape[1], other.shape[1]))
                return self._mul_sparse_matrix(other)
            # self is a row.
            elif self.shape[0] == 1 and self.shape[1] == other.shape[1]:
                copy = dia_matrix((self.toarray().ravel(), [0]),
                                  shape=(self.shape[1], self.shape[1]))
                return other._mul_sparse_matrix(copy)
            # Column vector times matrix. other is a column.
            elif other.shape[1] == 1 and self.shape[0] == other.shape[0]:
                other = dia_matrix((other.toarray().ravel(), [0]),
                                   shape=(other.shape[0], other.shape[0]))
                return other._mul_sparse_matrix(self)
            # self is a column.
            elif self.shape[1] == 1 and self.shape[0] == other.shape[0]:
                copy = dia_matrix((self.toarray().ravel(), [0]),
                                  shape=(self.shape[0], self.shape[0]))
                return copy._mul_sparse_matrix(other)
            else:
                raise ValueError("inconsistent shapes")

        # Assume other is a dense matrix/array, which produces a single-item
        # object array if other isn't convertible to ndarray.
        other = np.atleast_2d(other)

        if other.ndim != 2:
            return np.multiply(self.toarray(), other)
        # Single element / wrapped object.
        if other.size == 1:
            return self._mul_scalar(other.flat[0])
        # Fast case for trivial sparse matrix.
        elif self.shape == (1, 1):
            return np.multiply(self.toarray()[0, 0], other)

        ret = self.tocoo()
        # Matching shapes.
        if self.shape == other.shape:
            data = np.multiply(ret.data, other[ret.row, ret.col])
        # Sparse row vector times...
        elif self.shape[0] == 1:
            if other.shape[1] == 1:  # Dense column vector.
                data = np.multiply(ret.data, other)
            elif other.shape[1] == self.shape[1]:  # Dense matrix.
                data = np.multiply(ret.data, other[:, ret.col])
            else:
                raise ValueError("inconsistent shapes")
            row = np.repeat(np.arange(other.shape[0]), len(ret.row))
            col = np.tile(ret.col, other.shape[0])
            return coo_matrix((data.view(np.ndarray).ravel(), (row, col)),
                              shape=(other.shape[0], self.shape[1]),
                              copy=False)
        # Sparse column vector times...
        elif self.shape[1] == 1:
            if other.shape[0] == 1:  # Dense row vector.
                data = np.multiply(ret.data[:, None], other)
            elif other.shape[0] == self.shape[0]:  # Dense matrix.
                data = np.multiply(ret.data[:, None], other[ret.row])
            else:
                raise ValueError("inconsistent shapes")
            row = np.repeat(ret.row, other.shape[1])
            col = np.tile(np.arange(other.shape[1]), len(ret.col))
            return coo_matrix((data.view(np.ndarray).ravel(), (row, col)),
                              shape=(self.shape[0], other.shape[1]),
                              copy=False)
        # Sparse matrix times dense row vector.
        elif other.shape[0] == 1 and self.shape[1] == other.shape[1]:
            data = np.multiply(ret.data, other[:, ret.col].ravel())
        # Sparse matrix times dense column vector.
        elif other.shape[1] == 1 and self.shape[0] == other.shape[0]:
            data = np.multiply(ret.data, other[ret.row].ravel())
        else:
            raise ValueError("inconsistent shapes")
        ret.data = data.view(np.ndarray).ravel()
        return ret

    ###########################
    # Multiplication handlers #
    ###########################

    def _mul_vector(self, other):
        M, N = self.shape

        # output array
        result = np.zeros(M, dtype=upcast_char(self.dtype.char,
                                               other.dtype.char))

        # csr_matvec or csc_matvec
        fn = getattr(_sparsetools, self.format + '_matvec')
        fn(M, N, self.indptr, self.indices, self.data, other, result)

        return result

    def _mul_multivector(self, other):
        M, N = self.shape
        n_vecs = other.shape[1]  # number of column vectors

        result = np.zeros((M, n_vecs),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))

        # csr_matvecs or csc_matvecs
        fn = getattr(_sparsetools, self.format + '_matvecs')
        fn(M, N, n_vecs, self.indptr, self.indices, self.data,
           other.ravel(), result.ravel())

        return result

    def _mul_sparse_matrix(self, other):
        M, K1 = self.shape
        K2, N = other.shape

        major_axis = self._swap((M, N))[0]
        other = self.__class__(other)  # convert to this format

        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=M*N)
        indptr = np.empty(major_axis + 1, dtype=idx_dtype)

        fn = getattr(_sparsetools, self.format + '_matmat_pass1')
        fn(M, N,
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           indptr)

        nnz = indptr[-1]
        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=nnz)
        indptr = np.asarray(indptr, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

        fn = getattr(_sparsetools, self.format + '_matmat_pass2')
        fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        return self.__class__((data, indices, indptr), shape=(M, N))

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            raise ValueError("k exceeds matrix dimensions")
        fn = getattr(_sparsetools, self.format + "_diagonal")
        y = np.empty(min(rows + min(k, 0), cols - max(k, 0)),
                     dtype=upcast(self.dtype))
        fn(k, self.shape[0], self.shape[1], self.indptr, self.indices,
           self.data, y)
        return y

    diagonal.__doc__ = spmatrix.diagonal.__doc__

    #####################
    # Other binary ops  #
    #####################

    def _maximum_minimum(self, other, npop, op_name, dense_check):
        if isscalarlike(other):
            if dense_check(other):
                warn("Taking maximum (minimum) with > 0 (< 0) number results"
                     " to a dense matrix.", SparseEfficiencyWarning,
                     stacklevel=3)
                other_arr = np.empty(self.shape, dtype=np.asarray(other).dtype)
                other_arr.fill(other)
                other_arr = self.__class__(other_arr)
                return self._binopt(other_arr, op_name)
            else:
                self.sum_duplicates()
                new_data = npop(self.data, np.asarray(other))
                mat = self.__class__((new_data, self.indices, self.indptr),
                                     dtype=new_data.dtype, shape=self.shape)
                return mat
        elif isdense(other):
            return npop(self.todense(), other)
        elif isspmatrix(other):
            return self._binopt(other, op_name)
        else:
            raise ValueError("Operands not compatible.")

    def maximum(self, other):
        return self._maximum_minimum(other, np.maximum,
                                     '_maximum_', lambda x: np.asarray(x) > 0)

    maximum.__doc__ = spmatrix.maximum.__doc__

    def minimum(self, other):
        return self._maximum_minimum(other, np.minimum,
                                     '_minimum_', lambda x: np.asarray(x) < 0)

    minimum.__doc__ = spmatrix.minimum.__doc__

    #####################
    # Reduce operations #
    #####################

    def sum(self, axis=None, dtype=None, out=None):
        """Sum the matrix over the given axis.  If the axis is None, sum
        over both rows and columns, returning a scalar.
        """
        # The spmatrix base class already does axis=0 and axis=1 efficiently
        # so we only do the case axis=None here
        if (not hasattr(self, 'blocksize') and
                axis in self._swap(((1, -1), (0, 2)))[0]):
            # faster than multiplication for large minor axis in CSC/CSR
            res_dtype = get_sum_dtype(self.dtype)
            ret = np.zeros(len(self.indptr) - 1, dtype=res_dtype)

            major_index, value = self._minor_reduce(np.add)
            ret[major_index] = value
            ret = asmatrix(ret)
            if axis % 2 == 1:
                ret = ret.T

            if out is not None and out.shape != ret.shape:
                raise ValueError('dimensions do not match')

            return ret.sum(axis=(), dtype=dtype, out=out)
        # spmatrix will handle the remaining situations when axis
        # is in {None, -1, 0, 1}
        else:
            return spmatrix.sum(self, axis=axis, dtype=dtype, out=out)

    sum.__doc__ = spmatrix.sum.__doc__

    def _minor_reduce(self, ufunc, data=None):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.

        Warning: this does not call sum_duplicates()

        Returns
        -------
        major_index : array of ints
            Major indices where nonzero

        value : array of self.dtype
            Reduce result for nonzeros in each major_index
        """
        if data is None:
            data = self.data
        major_index = np.flatnonzero(np.diff(self.indptr))
        value = ufunc.reduceat(data,
                               downcast_intp_index(self.indptr[major_index]))
        return major_index, value

    #######################
    # Getting and Setting #
    #######################

    def _get_intXint(self, row, col):
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data,
            major, major + 1, minor, minor + 1)
        return data.sum(dtype=self.dtype)

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap((row, col))
        if major.step in (1, None) and minor.step in (1, None):
            return self._get_submatrix(major, minor, copy=True)
        return self._major_slice(major)._minor_slice(minor)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        idx_dtype = self.indices.dtype
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        major = np.asarray(major, dtype=idx_dtype)
        minor = np.asarray(minor, dtype=idx_dtype)

        val = np.empty(major.size, dtype=self.dtype)
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          major.size, major.ravel(), minor.ravel(), val)
        if major.ndim == 1:
            return asmatrix(val)
        return self.__class__(val.reshape(major.shape))

    def _get_columnXarray(self, row, col):
        # outer indexing
        major, minor = self._swap((row, col))
        return self._major_index_fancy(major)._minor_index_fancy(minor)

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        idx_dtype = self.indices.dtype
        indices = np.asarray(idx, dtype=idx_dtype).ravel()

        _, N = self._swap(self.shape)
        M = len(indices)
        new_shape = self._swap((M, N))
        if M == 0:
            return self.__class__(new_shape)

        row_nnz = np.diff(self.indptr)
        idx_dtype = self.indices.dtype
        res_indptr = np.zeros(M+1, dtype=idx_dtype)
        np.cumsum(row_nnz[idx], out=res_indptr[1:])

        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_row_index(M, indices, self.indptr, self.indices, self.data,
                      res_indices, res_data)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self.shape)
        start, stop, step = idx.indices(M)
        M = len(xrange(start, stop, step))
        new_shape = self._swap((M, N))
        if M == 0:
            return self.__class__(new_shape)

        row_nnz = np.diff(self.indptr)
        idx_dtype = self.indices.dtype
        res_indptr = np.zeros(M+1, dtype=idx_dtype)
        np.cumsum(row_nnz[idx], out=res_indptr[1:])

        if step == 1:
            all_idx = slice(self.indptr[start], self.indptr[stop])
            res_indices = np.array(self.indices[all_idx], copy=copy)
            res_data = np.array(self.data[all_idx], copy=copy)
        else:
            nnz = res_indptr[-1]
            res_indices = np.empty(nnz, dtype=idx_dtype)
            res_data = np.empty(nnz, dtype=self.dtype)
            csr_row_slice(start, stop, step, self.indptr, self.indices,
                          self.data, res_indices, res_data)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        idx_dtype = self.indices.dtype
        idx = np.asarray(idx, dtype=idx_dtype).ravel()

        M, N = self._swap(self.shape)
        k = len(idx)
        new_shape = self._swap((M, k))
        if k == 0:
            return self.__class__(new_shape)

        # pass 1: count idx entries and compute new indptr
        col_offsets = np.zeros(N, dtype=idx_dtype)
        res_indptr = np.empty_like(self.indptr)
        csr_column_index1(k, idx, M, N, self.indptr, self.indices,
                          col_offsets, res_indptr)

        # pass 2: copy indices/data for selected idxs
        col_order = np.argsort(idx).astype(idx_dtype, copy=False)
        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_column_index2(col_order, col_offsets, len(self.indices),
                          self.indices, self.data, res_indices, res_data)
        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self.shape)
        start, stop, step = idx.indices(N)
        N = len(xrange(start, stop, step))
        if N == 0:
            return self.__class__(self._swap((M, N)))
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        # TODO: don't fall back to fancy indexing here
        return self._minor_index_fancy(np.arange(start, stop, step))

    def _get_submatrix(self, major=None, minor=None, copy=False):
        """Return a submatrix of this matrix.

        major, minor: None, int, or slice with step 1
        """
        M, N = self._swap(self.shape)
        i0, i1 = _process_slice(major, M)
        j0, j1 = _process_slice(minor, N)

        if i0 == 0 and j0 == 0 and i1 == M and j1 == N:
            return self.copy() if copy else self

        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)

        shape = self._swap((i1 - i0, j1 - j0))
        return self.__class__((data, indices, indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _set_intXint(self, row, col, x):
        i, j = self._swap((row, col))
        self._set_many(i, j, x)

    def _set_arrayXarray(self, row, col, x):
        i, j = self._swap((row, col))
        self._set_many(i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # clear entries that will be overwritten
        self._zero_many(*self._swap((row, col)))

        M, N = row.shape  # matches col.shape
        broadcast_row = M != 1 and x.shape[0] == 1
        broadcast_col = N != 1 and x.shape[1] == 1
        r, c = x.row, x.col
        x = np.asarray(x.data, dtype=self.dtype)
        if broadcast_row:
            r = np.repeat(np.arange(M), len(r))
            c = np.tile(c, M)
            x = np.tile(x, M)
        if broadcast_col:
            r = np.repeat(r, N)
            c = np.tile(np.arange(N), len(c))
            x = np.repeat(x, N)
        # only assign entries in the new sparsity structure
        i, j = self._swap((row[r, c], col[r, c]))
        self._set_many(i, j, x)

    def _setdiag(self, values, k):
        if 0 in self.shape:
            return

        M, N = self.shape
        broadcast = (values.ndim == 0)

        if k < 0:
            if broadcast:
                max_index = min(M + k, N)
            else:
                max_index = min(M + k, N, len(values))
            i = np.arange(max_index, dtype=self.indices.dtype)
            j = np.arange(max_index, dtype=self.indices.dtype)
            i -= k

        else:
            if broadcast:
                max_index = min(M, N - k)
            else:
                max_index = min(M, N - k, len(values))
            i = np.arange(max_index, dtype=self.indices.dtype)
            j = np.arange(max_index, dtype=self.indices.dtype)
            j += k

        if not broadcast:
            values = values[:len(i)]

        self[i, j] = values

    def _prepare_indices(self, i, j):
        M, N = self._swap(self.shape)

        def check_bounds(indices, bound):
            idx = indices.max()
            if idx >= bound:
                raise IndexError('index (%d) out of range (>= %d)' %
                                 (idx, bound))
            idx = indices.min()
            if idx < -bound:
                raise IndexError('index (%d) out of range (< -%d)' %
                                 (idx, bound))

        i = np.array(i, dtype=self.indices.dtype, copy=False, ndmin=1).ravel()
        j = np.array(j, dtype=self.indices.dtype, copy=False, ndmin=1).ravel()
        check_bounds(i, M)
        check_bounds(j, N)
        return i, j, M, N

    def _set_many(self, i, j, x):
        """Sets value at each (i, j) to x

        Here (i,j) index major and minor respectively, and must not contain
        duplicate entries.
        """
        i, j, M, N = self._prepare_indices(i, j)
        x = np.array(x, dtype=self.dtype, copy=False, ndmin=1).ravel()

        n_samples = x.size
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # rinse and repeat
            self.sum_duplicates()
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)

        if -1 not in offsets:
            # only affects existing non-zero cells
            self.data[offsets] = x
            return

        else:
            warn("Changing the sparsity structure of a {}_matrix is expensive."
                 " lil_matrix is more efficient.".format(self.format),
                 SparseEfficiencyWarning, stacklevel=3)
            # replace where possible
            mask = offsets > -1
            self.data[offsets[mask]] = x[mask]
            # only insertions remain
            mask = ~mask
            i = i[mask]
            i[i < 0] += M
            j = j[mask]
            j[j < 0] += N
            self._insert_many(i, j, x[mask])

    def _zero_many(self, i, j):
        """Sets value at each (i, j) to zero, preserving sparsity structure.

        Here (i,j) index major and minor respectively.
        """
        i, j, M, N = self._prepare_indices(i, j)

        n_samples = len(i)
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # rinse and repeat
            self.sum_duplicates()
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)

        # only assign zeros to the existing sparsity structure
        self.data[offsets[offsets > -1]] = 0

    def _insert_many(self, i, j, x):
        """Inserts new nonzero at each (i, j) with value x

        Here (i,j) index major and minor respectively.
        i, j and x must be non-empty, 1d arrays.
        Inserts each major group (e.g. all entries per row) at a time.
        Maintains has_sorted_indices property.
        Modifies i, j, x in place.
        """
        order = np.argsort(i, kind='mergesort')  # stable for duplicates
        i = i.take(order, mode='clip')
        j = j.take(order, mode='clip')
        x = x.take(order, mode='clip')

        do_sort = self.has_sorted_indices

        # Update index data type
        idx_dtype = get_index_dtype((self.indices, self.indptr),
                                    maxval=(self.indptr[-1] + x.size))
        self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
        self.indices = np.asarray(self.indices, dtype=idx_dtype)
        i = np.asarray(i, dtype=idx_dtype)
        j = np.asarray(j, dtype=idx_dtype)

        # Collate old and new in chunks by major index
        indices_parts = []
        data_parts = []
        ui, ui_indptr = np.unique(i, return_index=True)
        ui_indptr = np.append(ui_indptr, len(j))
        new_nnzs = np.diff(ui_indptr)
        prev = 0
        for c, (ii, js, je) in enumerate(six.izip(ui, ui_indptr, ui_indptr[1:])):
            # old entries
            start = self.indptr[prev]
            stop = self.indptr[ii]
            indices_parts.append(self.indices[start:stop])
            data_parts.append(self.data[start:stop])

            # handle duplicate j: keep last setting
            uj, uj_indptr = np.unique(j[js:je][::-1], return_index=True)
            if len(uj) == je - js:
                indices_parts.append(j[js:je])
                data_parts.append(x[js:je])
            else:
                indices_parts.append(j[js:je][::-1][uj_indptr])
                data_parts.append(x[js:je][::-1][uj_indptr])
                new_nnzs[c] = len(uj)

            prev = ii

        # remaining old entries
        start = self.indptr[ii]
        indices_parts.append(self.indices[start:])
        data_parts.append(self.data[start:])

        # update attributes
        self.indices = np.concatenate(indices_parts)
        self.data = np.concatenate(data_parts)
        nnzs = np.empty(self.indptr.shape, dtype=idx_dtype)
        nnzs[0] = idx_dtype(0)
        indptr_diff = np.diff(self.indptr)
        indptr_diff[ui] += new_nnzs
        nnzs[1:] = indptr_diff
        self.indptr = np.cumsum(nnzs, out=nnzs)

        if do_sort:
            # TODO: only sort where necessary
            self.has_sorted_indices = False
            self.sort_indices()

        self.check_format(full_check=False)

    ######################
    # Conversion methods #
    ######################

    def tocoo(self, copy=True):
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        _sparsetools.expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))

        return coo_matrix((self.data, (row, col)), self.shape, copy=copy,
                          dtype=self.dtype)

    tocoo.__doc__ = spmatrix.tocoo.__doc__

    def toarray(self, order=None, out=None):
        if out is None and order is None:
            order = self._swap('cf')[0]
        out = self._process_toarray_args(order, out)
        if not (out.flags.c_contiguous or out.flags.f_contiguous):
            raise ValueError('Output array must be C or F contiguous')
        # align ideal order with output array order
        if out.flags.c_contiguous:
            x = self.tocsr()
            y = out
        else:
            x = self.tocsc()
            y = out.T
        M, N = x._swap(x.shape)
        csr_todense(M, N, x.indptr, x.indices, x.data, y)
        return out

    toarray.__doc__ = spmatrix.toarray.__doc__

    ##############################################################
    # methods that examine or modify the internal data structure #
    ##############################################################

    def eliminate_zeros(self):
        """Remove zero entries from the matrix

        This is an *in place* operation
        """
        M, N = self._swap(self.shape)
        _sparsetools.csr_eliminate_zeros(M, N, self.indptr, self.indices,
                                         self.data)
        self.prune()  # nnz may have changed

    def __get_has_canonical_format(self):
        """Determine whether the matrix has sorted indices and no duplicates

        Returns
            - True: if the above applies
            - False: otherwise

        has_canonical_format implies has_sorted_indices, so if the latter flag
        is False, so will the former be; if the former is found True, the
        latter flag is also set.
        """

        # first check to see if result was cached
        if not getattr(self, '_has_sorted_indices', True):
            # not sorted => not canonical
            self._has_canonical_format = False
        elif not hasattr(self, '_has_canonical_format'):
            self.has_canonical_format = _sparsetools.csr_has_canonical_format(
                len(self.indptr) - 1, self.indptr, self.indices)
        return self._has_canonical_format

    def __set_has_canonical_format(self, val):
        self._has_canonical_format = bool(val)
        if val:
            self.has_sorted_indices = True

    has_canonical_format = property(fget=__get_has_canonical_format,
                                    fset=__set_has_canonical_format)

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together

        The is an *in place* operation
        """
        if self.has_canonical_format:
            return
        self.sort_indices()

        M, N = self._swap(self.shape)
        _sparsetools.csr_sum_duplicates(M, N, self.indptr, self.indices,
                                        self.data)

        self.prune()  # nnz may have changed
        self.has_canonical_format = True

    def __get_sorted(self):
        """Determine whether the matrix has sorted indices

        Returns
            - True: if the indices of the matrix are in sorted order
            - False: otherwise

        """

        # first check to see if result was cached
        if not hasattr(self, '_has_sorted_indices'):
            self._has_sorted_indices = _sparsetools.csr_has_sorted_indices(
                len(self.indptr) - 1, self.indptr, self.indices)
        return self._has_sorted_indices

    def __set_sorted(self, val):
        self._has_sorted_indices = bool(val)

    has_sorted_indices = property(fget=__get_sorted, fset=__set_sorted)

    def sorted_indices(self):
        """Return a copy of this matrix with sorted indices
        """
        A = self.copy()
        A.sort_indices()
        return A

        # an alternative that has linear complexity is the following
        # although the previous option is typically faster
        # return self.toother().toother()

    def sort_indices(self):
        """Sort the indices of this matrix *in place*
        """

        if not self.has_sorted_indices:
            _sparsetools.csr_sort_indices(len(self.indptr) - 1, self.indptr,
                                          self.indices, self.data)
            self.has_sorted_indices = True

    def prune(self):
        """Remove empty space after all non-zero elements.
        """
        major_dim = self.shape[0]

        if len(self.indptr) != major_dim + 1:
            raise ValueError('index pointer has invalid length')
        if len(self.indices) < self.nnz:
            raise ValueError('indices array has fewer than nnz elements')
        if len(self.data) < self.nnz:
            raise ValueError('data array has fewer than nnz elements')

        self.indices = _prune_array(self.indices[:self.nnz])
        self.data = _prune_array(self.data[:self.nnz])

    def resize(self, *shape):
        shape = check_shape(shape)
        if hasattr(self, 'blocksize'):
            bm, bn = self.blocksize
            new_M, rm = divmod(shape[0], bm)
            new_N, rn = divmod(shape[1], bn)
            if rm or rn:
                raise ValueError("shape must be divisible into %s blocks. "
                                 "Got %s" % (self.blocksize, shape))
            M, N = self.shape[0] // bm, self.shape[1] // bn
        else:
            new_M, new_N = self._swap(shape)
            M, N = self._swap(self.shape)

        if new_M < M:
            self.indices = self.indices[:self.indptr[new_M]]
            self.data = self.data[:self.indptr[new_M]]
            self.indptr = self.indptr[:new_M + 1]
        elif new_M > M:
            self.indptr = np.resize(self.indptr, new_M + 1)
            self.indptr[M + 1:].fill(self.indptr[M])

        if new_N < N:
            mask = self.indices < new_N
            if not np.all(mask):
                self.indices = self.indices[mask]
                self.data = self.data[mask]
                major_index, val = self._minor_reduce(np.add, mask)
                self.indptr.fill(0)
                self.indptr[1:][major_index] = val
                np.cumsum(self.indptr, out=self.indptr)

        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

    ###################
    # utility methods #
    ###################

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            return self.__class__((data, self.indices.copy(),
                                   self.indptr.copy()),
                                  shape=self.shape,
                                  dtype=data.dtype)
        else:
            return self.__class__((data, self.indices, self.indptr),
                                  shape=self.shape, dtype=data.dtype)

    def _binopt(self, other, op):
        """apply the binary operation fn to two sparse matrices."""
        other = self.__class__(other)

        # e.g. csr_plus_csr, csr_minus_csr, etc.
        fn = getattr(_sparsetools, self.format + op + self.format)

        maxnnz = self.nnz + other.nnz
        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=maxnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(maxnnz, dtype=idx_dtype)

        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(maxnnz, dtype=np.bool_)
        else:
            data = np.empty(maxnnz, dtype=upcast(self.dtype, other.dtype))

        fn(self.shape[0], self.shape[1],
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        A = self.__class__((data, indices, indptr), shape=self.shape)
        A.prune()

        return A

    def _divide_sparse(self, other):
        """
        Divide this matrix by a second sparse matrix.
        """
        if other.shape != self.shape:
            raise ValueError('inconsistent shapes')

        r = self._binopt(other, '_eldiv_')

        if np.issubdtype(r.dtype, np.inexact):
            # Eldiv leaves entries outside the combined sparsity
            # pattern empty, so they must be filled manually.
            # Everything outside of other's sparsity is NaN, and everything
            # inside it is either zero or defined by eldiv.
            out = np.empty(self.shape, dtype=self.dtype)
            out.fill(np.nan)
            row, col = other.nonzero()
            out[row, col] = 0
            r = r.tocoo()
            out[r.row, r.col] = r.data
            out = matrix(out)
        else:
            # integers types go with nan <-> 0
            out = r

        return out

class csr_matrix(_cs_matrix):
    """
    Compressed Sparse Row matrix

    This can be instantiated in several ways:
        csr_matrix(D)
            with a dense matrix or rank-2 ndarray D

        csr_matrix(S)
            with another sparse matrix S (equivalent to S.tocsr())

        csr_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the matrix dimensions
            are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        CSR format data array of the matrix
    indices
        CSR format index array of the matrix
    indptr
        CSR format index pointer array of the matrix
    has_sorted_indices
        Whether indices are sorted

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> csr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    As an example of how to construct a CSR matrix incrementally,
    the following snippet builds a term-document matrix from texts:

    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    >>> indptr = [0]
    >>> indices = []
    >>> data = []
    >>> vocabulary = {}
    >>> for d in docs:
    ...     for term in d:
    ...         index = vocabulary.setdefault(term, len(vocabulary))
    ...         indices.append(index)
    ...         data.append(1)
    ...     indptr.append(len(indices))
    ...
    >>> csr_matrix((data, indices, indptr), dtype=int).toarray()
    array([[2, 1, 0, 0],
           [0, 1, 1, 1]])

    """
    format = 'csr'

class dok_matrix(spmatrix, IndexMixin, dict):
    """
    Dictionary Of Keys based sparse matrix.

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        dok_matrix(D)
            with a dense matrix, D

        dok_matrix(S)
            with a sparse matrix, S

        dok_matrix((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Allows for efficient O(1) access of individual elements.
    Duplicates are not allowed.
    Can be efficiently converted to a coo_matrix once constructed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import dok_matrix
    >>> S = dok_matrix((5, 5), dtype=np.float32)
    >>> for i in range(5):
    ...     for j in range(5):
    ...         S[i, j] = i + j    # Update element

    """
    format = 'dok'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        dict.__init__(self)
        spmatrix.__init__(self)

        self.dtype = getdtype(dtype, default=float)
        if isinstance(arg1, tuple) and isshape(arg1):  # (M,N)
            M, N = arg1
            self._shape = check_shape((M, N))
        elif isspmatrix(arg1):  # Sparse ctor
            if isspmatrix_dok(arg1) and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.todok()

            if dtype is not None:
                arg1 = arg1.astype(dtype)

            dict.update(self, arg1)
            self._shape = check_shape(arg1.shape)
            self.dtype = arg1.dtype
        else:  # Dense ctor
            try:
                arg1 = np.asarray(arg1)
            except Exception:
                raise TypeError('Invalid input format.')

            if len(arg1.shape) != 2:
                raise TypeError('Expected rank <=2 dense array or matrix.')

            d = coo_matrix(arg1, dtype=dtype).todok()
            dict.update(self, d)
            self._shape = check_shape(arg1.shape)
            self.dtype = d.dtype

    def update(self, val):
        # Prevent direct usage of update
        raise NotImplementedError("Direct modification to dok_matrix element "
                                  "is not allowed.")

    def _update(self, data):
        """An update method for dict data defined for direct access to
        `dok_matrix` data. Main purpose is to be used for effcient conversion
        from other spmatrix classes. Has no checking if `data` is valid."""
        return dict.update(self, data)

    def set_shape(self, shape):
        new_matrix = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_matrix.__dict__
        dict.clear(self)
        dict.update(self, new_matrix)

    shape = property(fget=spmatrix.get_shape, fset=set_shape)

    def getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("getnnz over an axis is not implemented "
                                      "for DOK format.")
        return dict.__len__(self)

    def count_nonzero(self):
        return sum(x != 0 for x in six.itervalues(self))

    getnnz.__doc__ = spmatrix.getnnz.__doc__
    count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__

    def __len__(self):
        return dict.__len__(self)

    def get(self, key, default=0.):
        """This overrides the dict.get method, providing type checking
        but otherwise equivalent functionality.
        """
        try:
            i, j = key
            assert isintlike(i) and isintlike(j)
        except (AssertionError, TypeError, ValueError):
            raise IndexError('Index must be a pair of integers.')
        if (i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]):
            raise IndexError('Index out of bounds.')
        return dict.get(self, key, default)

    def _get_intXint(self, row, col):
        return dict.get(self, (row, col), self.dtype.type(0))

    def _get_intXslice(self, row, col):
        return self._get_sliceXslice(slice(row, row+1), col)

    def _get_sliceXint(self, row, col):
        return self._get_sliceXslice(row, slice(col, col+1))

    def _get_sliceXslice(self, row, col):
        row_start, row_stop, row_step = row.indices(self.shape[0])
        col_start, col_stop, col_step = col.indices(self.shape[1])
        row_range = six.xrange(row_start, row_stop, row_step)
        col_range = six.xrange(col_start, col_stop, col_step)
        shape = (len(row_range), len(col_range))
        # Switch paths only when advantageous
        # (count the iterations in the loops, adjust for complexity)
        if len(self) >= 2 * shape[0] * shape[1]:
            # O(nr*nc) path: loop over <row x col>
            return self._get_columnXarray(row_range, col_range)
        # O(nnz) path: loop over entries of self
        newdok = dok_matrix(shape, dtype=self.dtype)
        for key in six.iterkeys(self):
            i, ri = divmod(int(key[0]) - row_start, row_step)
            if ri != 0 or i < 0 or i >= shape[0]:
                continue
            j, rj = divmod(int(key[1]) - col_start, col_step)
            if rj != 0 or j < 0 or j >= shape[1]:
                continue
            x = dict.__getitem__(self, key)
            dict.__setitem__(newdok, (i, j), x)
        return newdok

    def _get_intXarray(self, row, col):
        return self._get_columnXarray([row], col)

    def _get_arrayXint(self, row, col):
        return self._get_columnXarray(row, [col])

    def _get_sliceXarray(self, row, col):
        row = list(range(*row.indices(self.shape[0])))
        return self._get_columnXarray(row, col)

    def _get_arrayXslice(self, row, col):
        col = list(range(*col.indices(self.shape[1])))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        # outer indexing
        newdok = dok_matrix((len(row), len(col)), dtype=self.dtype)

        for i, r in enumerate(row):
            for j, c in enumerate(col):
                v = dict.get(self, (r, c), 0)
                if v:
                    dict.__setitem__(newdok, (i, j), v)
        return newdok

    def _get_arrayXarray(self, row, col):
        # inner indexing
        i, j = map(np.atleast_2d, np.broadcast_arrays(row, col))
        newdok = dok_matrix(i.shape, dtype=self.dtype)

        for key in itertools.product(xrange(i.shape[0]), six.xrange(i.shape[1])):
            v = dict.get(self, (i[key], j[key]), 0)
            if v:
                dict.__setitem__(newdok, key, v)
        return newdok

    def _set_intXint(self, row, col, x):
        key = (row, col)
        if x:
            dict.__setitem__(self, key, x)
        elif dict.__contains__(self, key):
            del self[key]

    def _set_arrayXarray(self, row, col, x):
        row = list(map(int, row.ravel()))
        col = list(map(int, col.ravel()))
        x = x.ravel()
        dict.update(self, six.izip(six.izip(row, col), x))

        for i in np.nonzero(x == 0)[0]:
            key = (row[i], col[i])
            if dict.__getitem__(self, key) == 0:
                # may have been superseded by later update
                del self[key]

    def __add__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = dok_matrix(self.shape, dtype=res_dtype)
            # Add this scalar to every element.
            M, N = self.shape
            for key in itertools.product(xrange(M), six.xrange(N)):
                aij = dict.get(self, (key), 0) + other
                if aij:
                    new[key] = aij
            # new.dtype.char = self.dtype.char
        elif isspmatrix_dok(other):
            if other.shape != self.shape:
                raise ValueError("Matrix dimensions are not equal.")
            # We could alternatively set the dimensions to the largest of
            # the two matrices to be summed.  Would this be a good idea?
            res_dtype = upcast(self.dtype, other.dtype)
            new = dok_matrix(self.shape, dtype=res_dtype)
            dict.update(new, self)
            with np.errstate(over='ignore'):
                dict.update(new,
                           ((k, new[k] + other[k]) for k in six.iterkeys(other)))
        elif isspmatrix(other):
            csc = self.tocsc()
            new = csc + other
        elif isdense(other):
            new = self.todense() + other
        else:
            return NotImplemented
        return new

    def __radd__(self, other):
        if isscalarlike(other):
            new = dok_matrix(self.shape, dtype=self.dtype)
            M, N = self.shape
            for key in itertools.product(xrange(M), six.xrange(N)):
                aij = dict.get(self, (key), 0) + other
                if aij:
                    new[key] = aij
        elif isspmatrix_dok(other):
            if other.shape != self.shape:
                raise ValueError("Matrix dimensions are not equal.")
            new = dok_matrix(self.shape, dtype=self.dtype)
            dict.update(new, self)
            dict.update(new,
                       ((k, self[k] + other[k]) for k in six.iterkeys(other)))
        elif isspmatrix(other):
            csc = self.tocsc()
            new = csc + other
        elif isdense(other):
            new = other + self.todense()
        else:
            return NotImplemented
        return new

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('Negating a sparse boolean matrix is not'
                                      ' supported.')
        new = dok_matrix(self.shape, dtype=self.dtype)
        dict.update(new, ((k, -self[k]) for k in six.iterkeys(self)))
        return new

    def _mul_scalar(self, other):
        res_dtype = upcast_scalar(self.dtype, other)
        # Multiply this scalar by every element.
        new = dok_matrix(self.shape, dtype=res_dtype)
        dict.update(new, ((k, v * other) for k, v in six.iteritems(self)))
        return new

    def _mul_vector(self, other):
        # matrix * vector
        result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))
        for (i, j), v in six.iteritems(self):
            result[i] += v * other[j]
        return result

    def _mul_multivector(self, other):
        # matrix * multivector
        result_shape = (self.shape[0], other.shape[1])
        result_dtype = upcast(self.dtype, other.dtype)
        result = np.zeros(result_shape, dtype=result_dtype)
        for (i, j), v in six.iteritems(self):
            result[i,:] += v * other[j,:]
        return result

    def __imul__(self, other):
        if isscalarlike(other):
            dict.update(self, ((k, v * other) for k, v in six.iteritems(self)))
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = dok_matrix(self.shape, dtype=res_dtype)
            dict.update(new, ((k, v / other) for k, v in six.iteritems(self)))
            return new
        return self.tocsr() / other

    def __itruediv__(self, other):
        if isscalarlike(other):
            dict.update(self, ((k, v / other) for k, v in six.iteritems(self)))
            return self
        return NotImplemented

    def __reduce__(self):
        # this approach is necessary because __setstate__ is called after
        # __setitem__ upon unpickling and since __init__ is not called there
        # is no shape attribute hence it is not possible to unpickle it.
        return dict.__reduce__(self)

    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  For now it returns the number
    # of non-zeros.

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'axes' parameter because swapping "
                             "dimensions is the only logical permutation.")

        M, N = self.shape
        new = dok_matrix((N, M), dtype=self.dtype, copy=copy)
        dict.update(new, (((right, left), val)
                          for (left, right), val in six.iteritems(self)))
        return new

    transpose.__doc__ = spmatrix.transpose.__doc__

    def conjtransp(self):
        """Return the conjugate transpose."""
        M, N = self.shape
        new = dok_matrix((N, M), dtype=self.dtype)
        dict.update(new, (((right, left), np.conj(val))
                          for (left, right), val in six.iteritems(self)))
        return new

    def copy(self):
        new = dok_matrix(self.shape, dtype=self.dtype)
        dict.update(new, self)
        return new

    copy.__doc__ = spmatrix.copy.__doc__

    def tocoo(self, copy=False):
        if self.nnz == 0:
            return coo_matrix(self.shape, dtype=self.dtype)

        idx_dtype = get_index_dtype(maxval=max(self.shape))
        data = np.fromiter(itervalues(self), dtype=self.dtype, count=self.nnz)
        row = np.fromiter((i for i, _ in six.iterkeys(self)), dtype=idx_dtype, count=self.nnz)
        col = np.fromiter((j for _, j in six.iterkeys(self)), dtype=idx_dtype, count=self.nnz)
        A = coo_matrix((data, (row, col)), shape=self.shape, dtype=self.dtype)
        A.has_canonical_format = True
        return A

    tocoo.__doc__ = spmatrix.tocoo.__doc__

    def todok(self, copy=False):
        if copy:
            return self.copy()
        return self

    todok.__doc__ = spmatrix.todok.__doc__

    def tocsc(self, copy=False):
        return self.tocoo(copy=False).tocsc(copy=copy)

    tocsc.__doc__ = spmatrix.tocsc.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        newM, newN = shape
        M, N = self.shape
        if newM < M or newN < N:
            # Remove all elements outside new dimensions
            for (i, j) in list(iterkeys(self)):
                if i >= newM or j >= newN:
                    del self[i, j]
        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

class coo_matrix(_data_matrix, _minmax_mixin):
    """
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            with a dense matrix D

        coo_matrix(S)
            with another sparse matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate indices
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate indices are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """
    format = 'coo'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        if isinstance(arg1, tuple):
            if isshape(arg1):
                M, N = arg1
                self._shape = check_shape((M, N))
                idx_dtype = get_index_dtype(maxval=max(M, N))
                self.row = np.array([], dtype=idx_dtype)
                self.col = np.array([], dtype=idx_dtype)
                self.data = np.array([], getdtype(dtype, default=float))
                self.has_canonical_format = True
            else:
                try:
                    obj, (row, col) = arg1
                except (TypeError, ValueError):
                    raise TypeError('invalid input format')

                if shape is None:
                    if len(row) == 0 or len(col) == 0:
                        raise ValueError('cannot infer dimensions from zero '
                                         'sized index arrays')
                    M = operator.index(np.max(row)) + 1
                    N = operator.index(np.max(col)) + 1
                    self._shape = check_shape((M, N))
                else:
                    # Use 2 steps to ensure shape has length 2.
                    M, N = shape
                    self._shape = check_shape((M, N))

                idx_dtype = get_index_dtype(maxval=max(self.shape))
                self.row = np.array(row, copy=copy, dtype=idx_dtype)
                self.col = np.array(col, copy=copy, dtype=idx_dtype)
                self.data = np.array(obj, copy=copy)
                self.has_canonical_format = False

        else:
            if isspmatrix(arg1):
                if isspmatrix_coo(arg1) and copy:
                    self.row = arg1.row.copy()
                    self.col = arg1.col.copy()
                    self.data = arg1.data.copy()
                    self._shape = check_shape(arg1.shape)
                else:
                    coo = arg1.tocoo()
                    self.row = coo.row
                    self.col = coo.col
                    self.data = coo.data
                    self._shape = check_shape(coo.shape)
                self.has_canonical_format = False
            else:
                #dense argument
                M = np.atleast_2d(np.asarray(arg1))

                if M.ndim != 2:
                    raise TypeError('expected dimension <= 2 array or matrix')

                self._shape = check_shape(M.shape)
                if shape is not None:
                    if check_shape(shape) != self._shape:
                        raise ValueError('inconsistent shapes: %s != %s' %
                                         (shape, self._shape))

                self.row, self.col = M.nonzero()
                self.data = M[self.row, self.col]
                self.has_canonical_format = True

        if dtype is not None:
            self.data = self.data.astype(dtype, copy=False)

        self._check()

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        nrows, ncols = self.shape

        if order == 'C':
            # Upcast to avoid overflows: the coo_matrix constructor
            # below will downcast the results to a smaller dtype, if
            # possible.
            dtype = get_index_dtype(maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))

            flat_indices = np.multiply(ncols, self.row, dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, shape[1])
        elif order == 'F':
            dtype = get_index_dtype(maxval=(nrows * max(0, ncols - 1) + max(0, nrows - 1)))

            flat_indices = np.multiply(nrows, self.col, dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, shape[0])
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        # Handle copy here rather than passing on to the constructor so that no
        # copy will be made of new_row and new_col regardless
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        return coo_matrix((new_data, (new_row, new_col)),
                          shape=shape, copy=False)

    reshape.__doc__ = spmatrix.reshape.__doc__

    def getnnz(self, axis=None):
        if axis is None:
            nnz = len(self.data)
            if nnz != len(self.row) or nnz != len(self.col):
                raise ValueError('row, column, and data array must all be the '
                                 'same length')

            if self.data.ndim != 1 or self.row.ndim != 1 or \
                    self.col.ndim != 1:
                raise ValueError('row, column, and data arrays must be 1-D')

            return int(nnz)

        if axis < 0:
            axis += 2
        if axis == 0:
            return np.bincount(downcast_intp_index(self.col),
                               minlength=self.shape[1])
        elif axis == 1:
            return np.bincount(downcast_intp_index(self.row),
                               minlength=self.shape[0])
        else:
            raise ValueError('axis out of bounds')

    getnnz.__doc__ = spmatrix.getnnz.__doc__

    def _check(self):
        """ Checks data structure for consistency """

        # index arrays should have integer data types
        if self.row.dtype.kind != 'i':
            warn("row index array has non-integer dtype (%s)  "
                    % self.row.dtype.name)
        if self.col.dtype.kind != 'i':
            warn("col index array has non-integer dtype (%s) "
                    % self.col.dtype.name)

        idx_dtype = get_index_dtype(maxval=max(self.shape))
        self.row = np.asarray(self.row, dtype=idx_dtype)
        self.col = np.asarray(self.col, dtype=idx_dtype)
        self.data = to_native(self.data)

        if self.nnz > 0:
            if self.row.max() >= self.shape[0]:
                raise ValueError('row index exceeds matrix dimensions')
            if self.col.max() >= self.shape[1]:
                raise ValueError('column index exceeds matrix dimensions')
            if self.row.min() < 0:
                raise ValueError('negative row index found')
            if self.col.min() < 0:
                raise ValueError('negative column index found')

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape
        return coo_matrix((self.data, (self.col, self.row)),
                          shape=(N, M), copy=copy)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        if new_M < M or new_N < N:
            mask = np.logical_and(self.row < new_M, self.col < new_N)
            if not mask.all():
                self.row = self.row[mask]
                self.col = self.col[mask]
                self.data = self.data[mask]

        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

    def toarray(self, order=None, out=None):
        """See the docstring for `spmatrix.toarray`."""
        B = self._process_toarray_args(order, out)
        fortran = int(B.flags.f_contiguous)
        if not fortran and not B.flags.c_contiguous:
            raise ValueError("Output array must be C or F contiguous")
        M,N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    B.ravel('A'), fortran)
        return B

    def tocsc(self, copy=False):
        """Convert this matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.nnz == 0:
            return csc_matrix(self.shape, dtype=self.dtype)
        else:
            M,N = self.shape
            idx_dtype = get_index_dtype((self.col, self.row),
                                        maxval=max(self.nnz, M))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(N + 1, dtype=idx_dtype)
            indices = np.empty_like(row, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(N, M, self.nnz, col, row, self.data,
                      indptr, indices, data)

            x = csc_matrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.nnz == 0:
            return csr_matrix(self.shape, dtype=self.dtype)
        else:
            M,N = self.shape
            idx_dtype = get_index_dtype((self.row, self.col),
                                        maxval=max(self.nnz, N))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty_like(col, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(M, N, self.nnz, row, col, self.data,
                      indptr, indices, data)

            x = csr_matrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocoo(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocoo.__doc__ = spmatrix.tocoo.__doc__

    def todia(self, copy=False):

        self.sum_duplicates()
        ks = self.col - self.row  # the diagonal for each nonzero
        diags, diag_idx = np.unique(ks, return_inverse=True)

        if len(diags) > 100:
            # probably undesired, should todia() have a maxdiags parameter?
            warn("Constructing a DIA matrix with %d diagonals "
                 "is inefficient" % len(diags), SparseEfficiencyWarning)

        #initialize and fill in data array
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data

        return dia_matrix((data,diags), shape=self.shape)

    todia.__doc__ = spmatrix.todia.__doc__

    def todok(self, copy=False):

        self.sum_duplicates()
        dok = dok_matrix((self.shape), dtype=self.dtype)
        dok._update( six.izip(six.izip(self.row,self.col),self.data))

        return dok

    todok.__doc__ = spmatrix.todok.__doc__

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            raise ValueError("k exceeds matrix dimensions")
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                        dtype=self.dtype)
        diag_mask = (self.row + k) == self.col

        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            row, _, data = self._sum_duplicates(self.row[diag_mask],
                                                self.col[diag_mask],
                                                self.data[diag_mask])
        diag[row + min(k, 0)] = data

        return diag

    diagonal.__doc__ = _data_matrix.diagonal.__doc__

    def _setdiag(self, values, k):
        M, N = self.shape
        if values.ndim and not len(values):
            return
        idx_dtype = self.row.dtype

        # Determine which triples to keep and where to put the new ones.
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M+k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N-k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)

        # Define the array of data consisting of the entries to be added.
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values

        # Update the internal structure.
        self.row = np.concatenate((self.row[keep], new_row))
        self.col = np.concatenate((self.col[keep], new_col))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    # needed by _data_matrix
    def _with_data(self,data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return coo_matrix((data, (self.row.copy(), self.col.copy())),
                                   shape=self.shape, dtype=data.dtype)
        else:
            return coo_matrix((data, (self.row, self.col)),
                                   shape=self.shape, dtype=data.dtype)

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together

        This is an *in place* operation
        """
        if self.has_canonical_format:
            return
        summed = self._sum_duplicates(self.row, self.col, self.data)
        self.row, self.col, self.data = summed
        self.has_canonical_format = True

    def _sum_duplicates(self, row, col, data):
        # Assumes (data, row, col) not in canonical format.
        if len(data) == 0:
            return row, col, data
        order = np.lexsort((row, col))
        row = row[order]
        col = col[order]
        data = data[order]
        unique_mask = ((row[1:] != row[:-1]) |
                       (col[1:] != col[:-1]))
        unique_mask = np.append(True, unique_mask)
        row = row[unique_mask]
        col = col[unique_mask]
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        return row, col, data

    def eliminate_zeros(self):
        """Remove zero entries from the matrix

        This is an *in place* operation
        """
        mask = self.data != 0
        self.data = self.data[mask]
        self.row = self.row[mask]
        self.col = self.col[mask]

    #######################
    # Arithmetic handlers #
    #######################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError('Incompatible shapes.')
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        M, N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    result.ravel('A'), fortran)
        return matrix(result, copy=False)

    def _mul_vector(self, other):
        #output array
        result = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
                                                            other.dtype.char))
        coo_matvec(self.nnz, self.row, self.col, self.data, other, result)
        return result

    def _mul_multivector(self, other):
        result = np.zeros((other.shape[1], self.shape[0]),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))
        for i, col in enumerate(other.T):
            coo_matvec(self.nnz, self.row, self.col, self.data, col, result[i])
        return result.T.view(type=type(other))

class dia_matrix(_data_matrix):
    """Sparse matrix with DIAgonal storage

    This can be instantiated in several ways:
        dia_matrix(D)
            with a dense matrix

        dia_matrix(S)
            with another sparse matrix S (equivalent to S.todia())

        dia_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_matrix((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        DIA format data array of the matrix
    offsets
        DIA format offset array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_matrix
    >>> dia_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_matrix((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    """
    format = 'dia'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        if isspmatrix_dia(arg1):
            if copy:
                arg1 = arg1.copy()
            self.data = arg1.data
            self.offsets = arg1.offsets
            self._shape = check_shape(arg1.shape)
        elif isspmatrix(arg1):
            if isspmatrix_dia(arg1) and copy:
                A = arg1.copy()
            else:
                A = arg1.todia()
            self.data = A.data
            self.offsets = A.offsets
            self._shape = check_shape(A.shape)
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self._shape = check_shape(arg1)
                self.data = np.zeros((0,0), getdtype(dtype, default=float))
                idx_dtype = get_index_dtype(maxval=max(self.shape))
                self.offsets = np.zeros((0), dtype=idx_dtype)
            else:
                try:
                    # Try interpreting it as (data, offsets)
                    data, offsets = arg1
                except Exception:
                    raise ValueError('unrecognized form for dia_matrix constructor')
                else:
                    if shape is None:
                        raise ValueError('expected a shape argument')
                    self.data = np.atleast_2d(np.array(arg1[0], dtype=dtype, copy=copy))
                    self.offsets = np.atleast_1d(np.array(arg1[1],
                                                          dtype=get_index_dtype(maxval=max(shape)),
                                                          copy=copy))
                    self._shape = check_shape(shape)
        else:
            #must be dense, convert to COO first, then to DIA
            try:
                arg1 = np.asarray(arg1)
            except Exception:
                raise ValueError("unrecognized form for"
                        " %s_matrix constructor" % self.format)
            A = coo_matrix(arg1, dtype=dtype, shape=shape).todia()
            self.data = A.data
            self.offsets = A.offsets
            self._shape = check_shape(A.shape)

        if dtype is not None:
            self.data = self.data.astype(dtype)

        #check format
        if self.offsets.ndim != 1:
            raise ValueError('offsets array must have rank 1')

        if self.data.ndim != 2:
            raise ValueError('data array must have rank 2')

        if self.data.shape[0] != len(self.offsets):
            raise ValueError('number of diagonals (%d) '
                    'does not match the number of offsets (%d)'
                    % (self.data.shape[0], len(self.offsets)))

        if len(np.unique(self.offsets)) != len(self.offsets):
            raise ValueError('offset array contains duplicate values')

    def __repr__(self):
        format = _formats[self.getformat()][1]
        return "<%dx%d sparse matrix of type '%s'\n" \
               "\twith %d stored elements (%d diagonals) in %s format>" % \
               (self.shape + (self.dtype.type, self.nnz, self.data.shape[0],
                              format))

    def _data_mask(self):
        """Returns a mask of the same shape as self.data, where
        mask[i,j] is True when data[i,j] corresponds to a stored element."""
        num_rows, num_cols = self.shape
        offset_inds = np.arange(self.data.shape[1])
        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        return mask

    def count_nonzero(self):
        mask = self._data_mask()
        return np.count_nonzero(self.data[mask])

    def getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("getnnz over an axis is not implemented "
                                      "for DIA format")
        M,N = self.shape
        nnz = 0
        for k in self.offsets:
            if k > 0:
                nnz += min(M,N-k)
            else:
                nnz += min(M+k,N)
        return int(nnz)

    getnnz.__doc__ = spmatrix.getnnz.__doc__
    count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__

    def sum(self, axis=None, dtype=None, out=None):
        validateaxis(axis)

        if axis is not None and axis < 0:
            axis += 2

        res_dtype = get_sum_dtype(self.dtype)
        num_rows, num_cols = self.shape
        ret = None

        if axis == 0:
            mask = self._data_mask()
            x = (self.data * mask).sum(axis=0)
            if x.shape[0] == num_cols:
                res = x
            else:
                res = np.zeros(num_cols, dtype=x.dtype)
                res[:x.shape[0]] = x
            ret = matrix(res, dtype=res_dtype)

        else:
            row_sums = np.zeros(num_rows, dtype=res_dtype)
            one = np.ones(num_cols, dtype=res_dtype)
            dia_matvec(num_rows, num_cols, len(self.offsets),
                       self.data.shape[1], self.offsets, self.data, one, row_sums)

            row_sums = matrix(row_sums)

            if axis is None:
                return row_sums.sum(dtype=dtype, out=out)

            if axis is not None:
                row_sums = row_sums.T

            ret = matrix(row_sums.sum(axis=axis))

        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=(), dtype=dtype, out=out)

    sum.__doc__ = spmatrix.sum.__doc__

    def _mul_vector(self, other):
        x = other

        y = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
                                                       x.dtype.char))

        L = self.data.shape[1]

        M,N = self.shape

        dia_matvec(M,N, len(self.offsets), L, self.offsets, self.data, x.ravel(), y.ravel())

        return y

    def _mul_multimatrix(self, other):
        return np.hstack([self._mul_vector(col).reshape(-1,1) for col in other.T])

    def _setdiag(self, values, k=0):
        M, N = self.shape

        if values.ndim == 0:
            # broadcast
            values_n = np.inf
        else:
            values_n = len(values)

        if k < 0:
            n = min(M + k, N, values_n)
            min_index = 0
            max_index = n
        else:
            n = min(M, N - k, values_n)
            min_index = k
            max_index = k + n

        if values.ndim != 0:
            # allow also longer sequences
            values = values[:n]

        if k in self.offsets:
            self.data[self.offsets == k, min_index:max_index] = values
        else:
            self.offsets = np.append(self.offsets, self.offsets.dtype.type(k))
            m = max(max_index, self.data.shape[1])
            data = np.zeros((self.data.shape[0]+1, m), dtype=self.data.dtype)
            data[:-1,:self.data.shape[1]] = self.data
            data[-1, min_index:max_index] = values
            self.data = data

    def todia(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    todia.__doc__ = spmatrix.todia.__doc__

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        num_rows, num_cols = self.shape
        max_dim = max(self.shape)

        # flip diagonal offsets
        offsets = -self.offsets

        # re-align the data matrix
        r = np.arange(len(offsets), dtype=np.intc)[:, None]
        c = np.arange(num_rows, dtype=np.intc) - (offsets % max_dim)[:, None]
        pad_amount = max(0, max_dim-self.data.shape[1])
        data = np.hstack((self.data, np.zeros((self.data.shape[0], pad_amount),
                                              dtype=self.data.dtype)))
        data = data[r, c]
        return dia_matrix((data, offsets), shape=(
            num_cols, num_rows), copy=copy)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            raise ValueError("k exceeds matrix dimensions")
        idx, = np.nonzero(self.offsets == k)
        first_col, last_col = max(0, k), min(rows + k, cols)
        if idx.size == 0:
            return np.zeros(last_col - first_col, dtype=self.data.dtype)
        return self.data[idx[0], first_col:last_col]

    diagonal.__doc__ = spmatrix.diagonal.__doc__

    def tocsc(self, copy=False):
        
        if self.nnz == 0:
            return csc_matrix(self.shape, dtype=self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = np.arange(offset_len)

        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)

        idx_dtype = get_index_dtype(maxval=max(self.shape))
        indptr = np.zeros(num_cols + 1, dtype=idx_dtype)
        indptr[1:offset_len+1] = np.cumsum(mask.sum(axis=0))
        indptr[offset_len+1:] = indptr[offset_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return csc_matrix((data, indices, indptr), shape=self.shape,
                          dtype=self.dtype)

    tocsc.__doc__ = spmatrix.tocsc.__doc__

    def tocoo(self, copy=False):
        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = np.arange(offset_len)

        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)
        row = row[mask]
        col = np.tile(offset_inds, num_offsets)[mask.ravel()]
        data = self.data[mask]

        A = coo_matrix((data,(row,col)), shape=self.shape, dtype=self.dtype)
        A.has_canonical_format = True
        return A

    tocoo.__doc__ = spmatrix.tocoo.__doc__

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays are copied.
        """
        if copy:
            return dia_matrix((data, self.offsets.copy()), shape=self.shape)
        else:
            return dia_matrix((data,self.offsets), shape=self.shape)

    def resize(self, *shape):
        shape = check_shape(shape)
        M, N = shape
        # we do not need to handle the case of expanding N
        self.data = self.data[:, :N]

        if (M > self.shape[0] and
                np.any(self.offsets + self.shape[0] < self.data.shape[1])):
            # explicitly clear values that were previously hidden
            mask = (self.offsets[:, None] + self.shape[0] <=
                    np.arange(self.data.shape[1]))
            self.data[mask] = 0

        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

class csc_matrix(_cs_matrix):
    """
    Compressed Sparse Column matrix

    This can be instantiated in several ways:

        csc_matrix(D)
            with a dense matrix or rank-2 ndarray D

        csc_matrix(S)
            with another sparse matrix S (equivalent to S.tocsc())

        csc_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csc_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSC representation where the row indices for
            column i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding values are stored in
            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
            not supplied, the matrix dimensions are inferred from
            the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        Data array of the matrix
    indices
        CSC format index array
    indptr
        CSC format index pointer array
    has_sorted_indices
        Whether indices are sorted

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)


    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> csc_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """
    format = 'csc'

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        return csr_matrix((self.data, self.indices,
                           self.indptr), (N, M), copy=copy)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def __iter__(self):
        for r in self.tocsr():
            yield r

    def tocsc(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocsc.__doc__ = spmatrix.tocsc.__doc__

    def tocsr(self, copy=False):
        M,N = self.shape
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csc_tocsr(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        A = csr_matrix((data, indices, indptr), shape=self.shape, copy=False)
        A.has_sorted_indices = True
        return A

    tocsr.__doc__ = spmatrix.tocsr.__doc__

    def nonzero(self):
        # CSC can't use _cs_matrix's .nonzero method because it
        # returns the indices sorted for self transposed.

        # Get row and col indices, from _cs_matrix.tocoo
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))

        # Remove explicit zeros
        nz_mask = self.data != 0
        row = row[nz_mask]
        col = col[nz_mask]

        # Sort them to be in C-style order
        ind = np.argsort(row, kind='mergesort')
        row = row[ind]
        col = col[ind]

        return row, col

    nonzero.__doc__ = _cs_matrix.nonzero.__doc__

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(minor=i).tocsr()

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSC matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(major=i, copy=True)

    def _get_intXarray(self, row, col):
        return self._major_index_fancy(col)._get_submatrix(minor=row)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._major_slice(col)._get_submatrix(minor=row)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._get_submatrix(major=col)._minor_slice(row)

    def _get_sliceXarray(self, row, col):
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        return self._get_submatrix(major=col)._minor_index_fancy(row)

    def _get_arrayXslice(self, row, col):
        return self._major_slice(col)._minor_index_fancy(row)

    # these functions are used by the parent class (_cs_matrix)
    # to remove redudancy between csc_matrix and csr_matrix
    def _swap(self, x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x[1], x[0]

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        
        return csc_matrix((self.data, self.indices,
                           self.indptr), shape=(N, M), copy=copy)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def tolil(self, copy=False):
        
        lil = lil_matrix(self.shape,dtype=self.dtype)

        self.sum_duplicates()
        ptr,ind,dat = self.indptr,self.indices,self.data
        rows, data = lil.rows, lil.data

        for n in six.xrange(self.shape[0]):
            start = ptr[n]
            end = ptr[n+1]
            rows[n] = ind[start:end].tolist()
            data[n] = dat[start:end].tolist()

        return lil

    tolil.__doc__ = spmatrix.tolil.__doc__

    def tocsr(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocsr.__doc__ = spmatrix.tocsr.__doc__

    def tocsc(self, copy=False):
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, self.shape[0]))
        indptr = np.empty(self.shape[1] + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csr_tocsc(self.shape[0], self.shape[1],
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        
        A = csc_matrix((data, indices, indptr), shape=self.shape)
        A.has_sorted_indices = True
        return A

    tocsc.__doc__ = spmatrix.tocsc.__doc__

    def tobsr(self, blocksize=None, copy=True):
        from .bsr import bsr_matrix

        if blocksize is None:
            from .spfuncs import estimate_blocksize
            return self.tobsr(blocksize=estimate_blocksize(self))

        elif blocksize == (1,1):
            arg1 = (self.data.reshape(-1,1,1),self.indices,self.indptr)
            return bsr_matrix(arg1, shape=self.shape, copy=copy)

        else:
            R,C = blocksize
            M,N = self.shape

            if R < 1 or C < 1 or M % R != 0 or N % C != 0:
                raise ValueError('invalid blocksize %s' % blocksize)

            blks = csr_count_blocks(M,N,R,C,self.indptr,self.indices)

            idx_dtype = get_index_dtype((self.indptr, self.indices),
                                        maxval=max(N//C, blks))
            indptr = np.empty(M//R+1, dtype=idx_dtype)
            indices = np.empty(blks, dtype=idx_dtype)
            data = np.zeros((blks,R,C), dtype=self.dtype)

            csr_tobsr(M, N, R, C,
                      self.indptr.astype(idx_dtype),
                      self.indices.astype(idx_dtype),
                      self.data,
                      indptr, indices, data.ravel())

            return bsr_matrix((data,indices,indptr), shape=self.shape)

    tobsr.__doc__ = spmatrix.tobsr.__doc__

    # these functions are used by the parent class (_cs_matrix)
    # to remove redudancy between csc_matrix and csr_matrix
    def _swap(self, x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x

    def __iter__(self):
        indptr = np.zeros(2, dtype=self.indptr.dtype)
        shape = (1, self.shape[1])
        i0 = 0
        for i1 in self.indptr[1:]:
            indptr[1] = i1 - i0
            indices = self.indices[i0:i1]
            data = self.data[i0:i1]
            yield csr_matrix((data, indices, indptr), shape=shape, copy=True)
            i0 = i1

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i, i + 1, 0, N)
        return csr_matrix((data, indices, indptr), shape=(1, N),
                          dtype=self.dtype, copy=False)

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, 0, M, i, i + 1)
        return csr_matrix((data, indices, indptr), shape=(M, 1),
                          dtype=self.dtype, copy=False)

    def _get_intXarray(self, row, col):
        return self.getrow(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        # TODO: uncomment this once it's faster:
        # return self.getrow(row)._minor_slice(col)

        M, N = self.shape
        start, stop, stride = col.indices(N)

        ii, jj = self.indptr[row:row+2]
        row_indices = self.indices[ii:jj]
        row_data = self.data[ii:jj]

        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)

        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0

        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = np.array([0, len(row_indices)])

        if stride < 0:
            row_data = row_data[::-1]
            row_indices = abs(row_indices[::-1])

        shape = (1, int(np.ceil(float(stop - start) / stride)))
        return csr_matrix((row_data, row_indices, row_indptr), shape=shape,
                          dtype=self.dtype, copy=False)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        return self._major_slice(row)._get_submatrix(minor=col)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            col = np.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        return self._major_index_fancy(row)._get_submatrix(minor=col)

class lil_matrix(spmatrix, IndexMixin):
    """Row-based list of lists sparse matrix

    This is a structure for constructing sparse matrices incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct a matrix efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_matrix(D)
            with a dense matrix or rank-2 ndarray D

        lil_matrix(S)
            with another sparse matrix S (equivalent to S.tolil())

        lil_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        LIL format data array of the matrix
    rows
        LIL format row index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse matrices
        - once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large matrices

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.


    """
    format = 'lil'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        spmatrix.__init__(self)
        self.dtype = getdtype(dtype, arg1, default=float)

        # First get the shape
        if isspmatrix(arg1):
            if isspmatrix_lil(arg1) and copy:
                A = arg1.copy()
            else:
                A = arg1.tolil()

            if dtype is not None:
                A = A.astype(dtype)

            self._shape = check_shape(A.shape)
            self.dtype = A.dtype
            self.rows = A.rows
            self.data = A.data
        elif isinstance(arg1,tuple):
            if isshape(arg1):
                if shape is not None:
                    raise ValueError('invalid use of shape parameter')
                M, N = arg1
                self._shape = check_shape((M, N))
                self.rows = np.empty((M,), dtype=object)
                self.data = np.empty((M,), dtype=object)
                for i in range(M):
                    self.rows[i] = []
                    self.data[i] = []
            else:
                raise TypeError('unrecognized lil_matrix constructor usage')
        else:
            # assume A is dense
            try:
                A = asmatrix(arg1)
            except TypeError:
                raise TypeError('unsupported matrix type')
            else:
                
                A = csr_matrix(A, dtype=dtype).tolil()

                self._shape = check_shape(A.shape)
                self.dtype = A.dtype
                self.rows = A.rows
                self.data = A.data

    def __iadd__(self,other):
        self[:,:] = self + other
        return self

    def __isub__(self,other):
        self[:,:] = self - other
        return self

    def __imul__(self,other):
        if isscalarlike(other):
            self[:,:] = self * other
            return self
        else:
            return NotImplemented

    def __itruediv__(self,other):
        if isscalarlike(other):
            self[:,:] = self / other
            return self
        else:
            return NotImplemented

    # Whenever the dimensions change, empty lists should be created for each
    # row

    def getnnz(self, axis=None):
        if axis is None:
            return sum([len(rowvals) for rowvals in self.data])
        if axis < 0:
            axis += 2
        if axis == 0:
            out = np.zeros(self.shape[1], dtype=np.intp)
            for row in self.rows:
                out[row] += 1
            return out
        elif axis == 1:
            return np.array([len(rowvals) for rowvals in self.data], dtype=np.intp)
        else:
            raise ValueError('axis out of bounds')

    def count_nonzero(self):
        return sum(np.count_nonzero(rowvals) for rowvals in self.data)

    getnnz.__doc__ = spmatrix.getnnz.__doc__
    count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__

    def __str__(self):
        val = ''
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                val += "  %s\t%s\n" % (str((i, j)), str(self.data[i][pos]))
        return val[:-1]

    def getrowview(self, i):
        """Returns a view of the 'i'th row (without copying).
        """
        new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i]
        new.data[0] = self.data[i]
        return new

    def getrow(self, i):
        """Returns a copy of the 'i'th row.
        """
        M, N = self.shape
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('row index out of bounds')
        new = lil_matrix((1, N), dtype=self.dtype)
        new.rows[0] = self.rows[i][:]
        new.data[0] = self.data[i][:]
        return new

    def __getitem__(self, key):
        # Fast path for simple (int, int) indexing.
        if (isinstance(key, tuple) and len(key) == 2 and
                isinstance(key[0], INT_TYPES) and
                isinstance(key[1], INT_TYPES)):
            # lil_get1 handles validation for us.
            return self._get_intXint(*key)
        # Everything else takes the normal path.
        return IndexMixin.__getitem__(self, key)

    def _asindices(self, idx, N):
        # LIL routines handle bounds-checking for us, so don't do it here.
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError):
            raise IndexError('invalid index')
        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')
        return x

    def _get_intXint(self, row, col):
        v = _csparsetools.lil_get1(self.shape[0], self.shape[1], self.rows,
                                   self.data, row, col)
        return self.dtype.type(v)

    def _get_sliceXint(self, row, col):
        row = six.xrange(*row.indices(self.shape[0]))
        return self._get_row_ranges(row, slice(col, col+1))

    def _get_arrayXint(self, row, col):
        return self._get_row_ranges(row, slice(col, col+1))

    def _get_intXslice(self, row, col):
        return self._get_row_ranges((row,), col)

    def _get_sliceXslice(self, row, col):
        row = six.xrange(*row.indices(self.shape[0]))
        return self._get_row_ranges(row, col)

    def _get_arrayXslice(self, row, col):
        return self._get_row_ranges(row, col)

    def _get_intXarray(self, row, col):
        row = np.array(row, dtype=col.dtype, ndmin=1)
        return self._get_columnXarray(row, col)

    def _get_sliceXarray(self, row, col):
        row = np.arange(*row.indices(self.shape[0]))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        # outer indexing
        row, col = _broadcast_arrays(row[:,None], col)
        return self._get_arrayXarray(row, col)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        i, j = map(np.atleast_2d, _prepare_index_for_memoryview(row, col))
        new = lil_matrix(i.shape, dtype=self.dtype)
        _csparsetools.lil_fancy_get(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    new.rows, new.data,
                                    i, j)
        return new

    def _get_row_ranges(self, rows, col_slice):
        """
        Fast path for indexing in the case where column index is slice.

        This gains performance improvement over brute force by more
        efficient skipping of zeros, by accessing the elements
        column-wise in order.

        Parameters
        ----------
        rows : sequence or six.xrange
            Rows indexed. If six.xrange, must be within valid bounds.
        col_slice : slice
            Columns indexed

        """
        j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
        col_range = six.xrange(j_start, j_stop, j_stride)
        nj = len(col_range)
        new = lil_matrix((len(rows), nj), dtype=self.dtype)

        _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1],
                                         self.rows, self.data,
                                         new.rows, new.data,
                                         rows,
                                         j_start, j_stop, j_stride, nj)

        return new

    def _set_intXint(self, row, col, x):
        _csparsetools.lil_insert(self.shape[0], self.shape[1], self.rows,
                                 self.data, row, col, x)

    def _set_arrayXarray(self, row, col, x):
        i, j, x = map(np.atleast_2d, _prepare_index_for_memoryview(row, col, x))
        _csparsetools.lil_fancy_set(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # Special case: full matrix assignment
        if (x.shape == self.shape and
                isinstance(row, slice) and row == slice(None) and
                isinstance(col, slice) and col == slice(None)):
            x = lil_matrix(x, dtype=self.dtype)
            self.rows = x.rows
            self.data = x.data
            return
        # Fall back to densifying x
        x = np.asarray(x.toarray(), dtype=self.dtype)
        x, _ = _broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)

    def __setitem__(self, key, x):
        # Fast path for simple (int, int) indexing.
        if (isinstance(key, tuple) and len(key) == 2 and
                isinstance(key[0], INT_TYPES) and
                isinstance(key[1], INT_TYPES)):
            x = self.dtype.type(x)
            if x.size > 1:
                raise ValueError("Trying to assign a sequence to an item")
            return self._set_intXint(key[0], key[1], x)
        # Everything else takes the normal path.
        IndexMixin.__setitem__(self, key, x)

    def _mul_scalar(self, other):
        if other == 0:
            # Multiply by zero: return the zero matrix
            new = lil_matrix(self.shape, dtype=self.dtype)
        else:
            res_dtype = upcast_scalar(self.dtype, other)

            new = self.copy()
            new = new.astype(res_dtype)
            # Multiply this scalar by every element.
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val*other for val in rowvals]
        return new

    def __truediv__(self, other):           # self / other
        if isscalarlike(other):
            new = self.copy()
            # Divide every element by this scalar
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val/other for val in rowvals]
            return new
        else:
            return self.tocsr() / other

    def copy(self):
        M, N = self.shape
        new = lil_matrix(self.shape, dtype=self.dtype)
        # This is ~14x faster than calling deepcopy() on rows and data.
        _csparsetools.lil_get_row_ranges(M, N, self.rows, self.data,
                                         new.rows, new.data, six.xrange(M),
                                         0, N, 1, N)
        return new

    copy.__doc__ = spmatrix.copy.__doc__

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        new = lil_matrix(shape, dtype=self.dtype)

        if order == 'C':
            ncols = self.shape[1]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    new_r, new_c = np.unravel_index(i * ncols + j, shape)
                    new[new_r, new_c] = self[i, j]
        elif order == 'F':
            nrows = self.shape[0]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    new_r, new_c = np.unravel_index(i + j * nrows, shape, order)
                    new[new_r, new_c] = self[i, j]
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        return new

    reshape.__doc__ = spmatrix.reshape.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        if new_M < M:
            self.rows = self.rows[:new_M]
            self.data = self.data[:new_M]
        elif new_M > M:
            self.rows = np.resize(self.rows, new_M)
            self.data = np.resize(self.data, new_M)
            for i in range(M, new_M):
                self.rows[i] = []
                self.data[i] = []

        if new_N < N:
            for row, data in zip(self.rows, self.data):
                trunc = bisect_left(row, new_N)
                del row[trunc:]
                del data[trunc:]

        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

    def toarray(self, order=None, out=None):
        d = self._process_toarray_args(order, out)
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                d[i, j] = self.data[i][pos]
        return d

    toarray.__doc__ = spmatrix.toarray.__doc__

    def transpose(self, axes=None, copy=False):
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False).tolil(copy=False)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def tolil(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tolil.__doc__ = spmatrix.tolil.__doc__

    def tocsr(self, copy=False):
        # construct indptr array
        M, N = self.shape
        lengths = np.fromiter(map(len, self.rows),
                              dtype=get_index_dtype(maxval=N), count=M)
        nnz = lengths.sum()
        idx_dtype = get_index_dtype(maxval=max(N, nnz))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indptr[0] = 0
        np.cumsum(lengths, dtype=idx_dtype, out=indptr[1:])
        # construct indices and data array
        # using faster construction approach depending on density
        # see https://github.com/scipy/scipy/pull/10939 for details
        if M == 0:
            indices = np.empty(0, dtype=idx_dtype)
            data = np.empty(0, dtype=self.dtype)
        elif nnz / M > 30:
            indices = np.empty(nnz, dtype=idx_dtype)
            data = np.empty(nnz, dtype=self.dtype)
            start = 0
            for i, stop in enumerate(indptr[1:]):
                if stop > start:
                    indices[start:stop] = self.rows[i]
                    data[start:stop] = self.data[i]
                    start = stop
        else:
            indices = np.fromiter((x for y in self.rows for x in y),
                                  dtype=idx_dtype, count=nnz)
            data = np.fromiter((x for y in self.data for x in y),
                               dtype=self.dtype, count=nnz)
        # init csr matrix
        
        return csr_matrix((data, indices, indptr), shape=self.shape)

    tocsr.__doc__ = spmatrix.tocsr.__doc__

def _process_slice(sl, num):
    if sl is None:
        i0, i1 = 0, num
    elif isinstance(sl, slice):
        i0, i1, stride = sl.indices(num)
        if stride != 1:
            raise ValueError('slicing with step != 1 not supported')
        i0 = min(i0, i1)  # give an empty slice when i0 > i1
    elif isintlike(sl):
        if sl < 0:
            sl += num
        i0, i1 = sl, sl + 1
        if i0 < 0 or i1 > num:
            raise IndexError('index out of bounds: 0 <= %d < %d <= %d' %
                             (i0, i1, num))
    else:
        raise TypeError('expected slice or scalar')

    return i0, i1

def isspmatrix_dia(x):
    """Is x of dia_matrix type?

    Parameters
    ----------
    x
        object to check for being a dia matrix

    Returns
    -------
    bool
        True if x is a dia matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dia_matrix, isspmatrix_dia
    >>> isspmatrix_dia(dia_matrix([[5]]))
    True

    >>> from scipy.sparse import dia_matrix, csr_matrix, isspmatrix_dia
    >>> isspmatrix_dia(csr_matrix([[5]]))
    False
    """
    return isinstance(x, dia_matrix)

def isspmatrix_dok(x):
    """Is x of dok_matrix type?

    Parameters
    ----------
    x
        object to check for being a dok matrix

    Returns
    -------
    bool
        True if x is a dok matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dok_matrix, isspmatrix_dok
    >>> isspmatrix_dok(dok_matrix([[5]]))
    True

    >>> from scipy.sparse import dok_matrix, csr_matrix, isspmatrix_dok
    >>> isspmatrix_dok(csr_matrix([[5]]))
    False
    """
    return isinstance(x, dok_matrix)

def isspmatrix_coo(x):
    """Is x of coo_matrix type?

    Parameters
    ----------
    x
        object to check for being a coo matrix

    Returns
    -------
    bool
        True if x is a coo matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, isspmatrix_coo
    >>> isspmatrix_coo(coo_matrix([[5]]))
    True

    >>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo
    >>> isspmatrix_coo(csr_matrix([[5]]))
    False
    """
    return isinstance(x, coo_matrix)

def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    if arr.dtype.itemsize > np.dtype(np.intp).itemsize:
        if arr.size == 0:
            return arr.astype(np.intp)
        maxval = arr.max()
        minval = arr.min()
        if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:
            raise ValueError("Cannot deal with arrays with indices larger "
                             "than the machine maximum address size "
                             "(e.g. 64-bit indices on 32-bit machine).")
        return arr.astype(np.intp)
    return arr

def check_shape(args, current_shape=None):
    """Imitate numpy.matrix handling of shape arguments"""
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: "
                        "'shape'")
    elif len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]), )
        else:
            new_shape = tuple(operator.index(arg) for arg in shape_iter)
    else:
        new_shape = tuple(operator.index(arg) for arg in args)

    if current_shape is None:
        if len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        elif new_shape[0] < 0 or new_shape[1] < 0:
            raise ValueError("'shape' elements cannot be negative")

    else:
        # Check the current size only if needed
        current_size = np.prod(current_shape, dtype=int)

        # Check for negatives
        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if len(negative_indexes) == 0:
            new_size = np.prod(new_shape, dtype=int)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 .format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = np.prod(new_shape[0:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape' if x < 0 else x for x in new_shape)
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 ''.format(current_size, err_shape))
            new_shape = new_shape[0:skip] + (unspecified,) + new_shape[skip+1:]
        else:
            raise ValueError('can only specify one unknown dimension')

    if len(new_shape) != 2:
        raise ValueError('matrix shape must be two-dimensional')

    return new_shape

def check_reshape_kwargs(kwargs):
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """

    order = kwargs.pop('order', 'C')
    copy = kwargs.pop('copy', False)
    if kwargs:  # Some unused kwargs remain
        raise TypeError('reshape() got unexpected keywords arguments: {}'
                        .format(', '.join(kwargs.keys())))
    return order, copy

def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False

def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w

def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1

def general_cosine(M, a, sym=True):
    r"""
    Generic weighted sum of cosine terms window

    Parameters
    ----------
    M : int
        Number of points in the output window
    a : array_like
        Sequence of weighting coefficients. This uses the convention of being
        centered on the origin, so these will typically all be positive
        numbers, not alternating sign.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf

    Examples
    --------
    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_

    .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
               - 0.440811 \cos(3z) + 0.043097 \cos(4z)

    where

    .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1

    Since this uses the convention of starting at the origin, to reproduce the
    window, we need to convert every other coefficient to a positive number:

    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
    Figure 42 by plotting the window and its frequency response, and confirm
    the sidelobe level in red:

    >>> from scipy.signal.windows import general_cosine
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = general_cosine(1000, HFT90D, sym=False)
    >>> plt.plot(window)
    >>> plt.title("HFT90D window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 10000) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * np.log10(np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-50/1000, 50/1000, -140, 0])
    >>> plt.title("Frequency response of the HFT90D window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axhline(-90.2, color='red')
    >>> plt.show()
    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return _truncate(w, needs_trunc)

def general_hamming(M, alpha, sym=True):
    r"""Return a generalized Hamming window.

    The generalized Hamming window is constructed by multiplying a rectangular
    window by one period of a cosine function [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float
        The window coefficient, :math:`\alpha`
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Hamming window is defined as

    .. math:: w(n) = \alpha - \left(1 - \alpha\right) \cos\left(\frac{2\pi{n}}{M-1}\right)
              \qquad 0 \leq n \leq M-1

    Both the common Hamming window and Hann window are special cases of the
    generalized Hamming window with :math:`\alpha` = 0.54 and :math:`\alpha` =
    0.5, respectively [2]_.

    See Also
    --------
    hamming, hann

    Examples
    --------
    The Sentinel-1A/B Instrument Processing Facility uses generalized Hamming
    windows in the processing of spaceborne Synthetic Aperture Radar (SAR)
    data [3]_. The facility uses various values for the :math:`\alpha`
    parameter based on operating mode of the SAR instrument. Some common
    :math:`\alpha` values include 0.75, 0.7 and 0.52 [4]_. As an example, we
    plot these different windows.

    >>> from scipy.signal.windows import general_hamming
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> fig1, spatial_plot = plt.subplots()
    >>> spatial_plot.set_title("Generalized Hamming Windows")
    >>> spatial_plot.set_ylabel("Amplitude")
    >>> spatial_plot.set_xlabel("Sample")

    >>> fig2, freq_plot = plt.subplots()
    >>> freq_plot.set_title("Frequency Responses")
    >>> freq_plot.set_ylabel("Normalized magnitude [dB]")
    >>> freq_plot.set_xlabel("Normalized frequency [cycles per sample]")

    >>> for alpha in [0.75, 0.7, 0.52]:
    ...     window = general_hamming(41, alpha)
    ...     spatial_plot.plot(window, label="{:.2f}".format(alpha))
    ...     A = fft(window, 2048) / (len(window)/2.0)
    ...     freq = np.linspace(-0.5, 0.5, len(A))
    ...     response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    ...     freq_plot.plot(freq, response, label="{:.2f}".format(alpha))
    >>> freq_plot.legend(loc="upper right")
    >>> spatial_plot.legend(loc="upper right")

    References
    ----------
    .. [1] DSPRelated, "Generalized Hamming Window Family",
           https://www.dsprelated.com/freebooks/sasp/Generalized_Hamming_Window_Family.html
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [3] Riccardo Piantanida ESA, "Sentinel-1 Level 1 Detailed Algorithm
           Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition
    .. [4] Matthieu Bourbigot ESA, "Sentinel-1 Product Definition",
           https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Definition
    """
    return general_cosine(M, [alpha, 1. - alpha], sym)

def kaiser(M, beta, sym=True):
    r"""Return a Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple approximation
    to the DPSS window based on Bessel functions.
    The Kaiser window is a very good approximation to the Digital Prolate
    Spheroidal Sequence, or Slepian window, which is the transform which
    maximizes the energy in the main lobe of the window relative to total
    energy.

    The Kaiser can approximate other windows by varying the beta parameter.
    (Some literature uses alpha = beta/pi.) [4]_

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hann
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    be returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transform," Proceedings of the IEEE, vol. 66,
           no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.kaiser(51, beta=14)
    >>> plt.plot(window)
    >>> plt.title(r"Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's kaiser function
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M)
    alpha = (M - 1) / 2.0
    w = (special.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         special.i0(beta))

    return _truncate(w, needs_trunc)

def hann(M, sym=True):
    r"""
    Return a Hann window.

    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The Hann window is defined as

    .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The window was named for Julius von Hann, an Austrian meteorologist. It is
    also known as the Cosine Bell. It is sometimes erroneously referred to as
    the "Hanning" window, from the use of "hann" as a verb in the original
    paper and confusion with the very similar Hamming window.

    Most references to the Hann window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.hann(51)
    >>> plt.plot(window)
    >>> plt.title("Hann window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * np.log10(np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Hann window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    # Docstring adapted from NumPy's hanning function
    return general_hamming(M, 0.5, sym)

_win_equiv_raw = {
    ('kaiser', 'ksr'): (kaiser, True),
    ('hanning', 'hann', 'han'): (hann, False)
}

# Fill dict with all valid window name strings
_win_equiv = {}
for k, v in _win_equiv_raw.items():
    for key in k:
        _win_equiv[key] = v[0]

_needs_param = set()
for k, v in _win_equiv_raw.items():
    if v[1]:
        _needs_param.update(k)

def scipy_get_window(window, Nx, fftbins=True):
    """
    Return a window of a given length and type.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to create. See below for more details.
    Nx : int
        The number of samples in the window.
    fftbins : bool, optional
        If True (default), create a "periodic" window, ready to use with
        `ifftshift` and be multiplied by the result of an FFT (see also
        :func:`~scipy.fft.fftfreq`).
        If False, create a "symmetric" window, for use in filter design.

    Returns
    -------
    get_window : ndarray
        Returns a window of length `Nx` and type `window`

    Notes
    -----
    Window types:

    - `~scipy.signal.windows.boxcar`
    - `~scipy.signal.windows.triang`
    - `~scipy.signal.windows.blackman`
    - `~scipy.signal.windows.hamming`
    - `~scipy.signal.windows.hann`
    - `~scipy.signal.windows.bartlett`
    - `~scipy.signal.windows.flattop`
    - `~scipy.signal.windows.parzen`
    - `~scipy.signal.windows.bohman`
    - `~scipy.signal.windows.blackmanharris`
    - `~scipy.signal.windows.nuttall`
    - `~scipy.signal.windows.barthann`
    - `~scipy.signal.windows.kaiser` (needs beta)
    - `~scipy.signal.windows.gaussian` (needs standard deviation)
    - `~scipy.signal.windows.general_gaussian` (needs power, width)
    - `~scipy.signal.windows.slepian` (needs width)
    - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
    - `~scipy.signal.windows.chebwin` (needs attenuation)
    - `~scipy.signal.windows.exponential` (needs decay scale)
    - `~scipy.signal.windows.tukey` (needs taper fraction)

    If the window requires no parameters, then `window` can be a string.

    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    If `window` is a floating point number, it is interpreted as the beta
    parameter of the `~scipy.signal.windows.kaiser` window.

    Each of the window types listed above is also the name of
    a function that can be called directly to create a window of
    that type.

    Examples
    --------
    >>> from scipy import signal
    >>> signal.get_window('triang', 7)
    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
    >>> signal.get_window(('kaiser', 4.0), 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])
    >>> signal.get_window(4.0, 9)
    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
            0.97885093,  0.82160913,  0.56437221,  0.29425961])

    """
    sym = not fftbins
    try:
        beta = float(window)
    except (TypeError, ValueError):
        args = ()
        if isinstance(window, tuple):
            winstr = window[0]
            if len(window) > 1:
                args = window[1:]
        elif isinstance(window, six.string_types):
            if window in _needs_param:
                raise ValueError("The '" + window + "' window needs one or "
                                 "more parameters -- pass a tuple.")
            else:
                winstr = window
        else:
            raise ValueError("%s as window type is not supported." %
                             str(type(window)))

        try:
            winfunc = _win_equiv[winstr]
        except KeyError:
            raise ValueError("Unknown window type.")

        params = (Nx,) + args + (sym,)
    else:
        winfunc = kaiser
        params = (Nx, beta, sym)

    return winfunc(*params)

def isspmatrix_lil(x):
    """Is x of lil_matrix type?

    Parameters
    ----------
    x
        object to check for being a lil matrix

    Returns
    -------
    bool
        True if x is a lil matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import lil_matrix, isspmatrix_lil
    >>> isspmatrix_lil(lil_matrix([[5]]))
    True

    >>> from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_lil
    >>> isspmatrix_lil(csr_matrix([[5]]))
    False
    """
    return isinstance(x, lil_matrix)

def _broadcast_arrays(a, b):
    """
    Same as np.broadcast_arrays(a, b) but old writeability rules.

    Numpy >= 1.17.0 transitions broadcast_arrays to return
    read-only arrays. Set writeability explicitly to avoid warnings.
    Retain the old writeability rules, as our Cython code assumes
    the old behavior.
    """
    x, y = np.broadcast_arrays(a, b)
    x.flags.writeable = a.flags.writeable
    y.flags.writeable = b.flags.writeable
    return x, y

# end scipy -------------------------------------------------------------------

# Util ------------------------------------------------------------------------
MAX_MEM_BLOCK = 2**8 * 2**10

class Deprecated(object):
    '''A dummy class to catch usage of deprecated variable names'''
    def __repr__(self):
        return '<DEPRECATED parameter>'

class LibrosaError(Exception):
    '''The root librosa exception class'''
    pass

class ParameterError(LibrosaError):
    '''Exception class for mal-formed inputs'''
    pass

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

def valid_audio(y, mono=True):
    '''Validate whether a variable contains valid, mono audio data.


    Parameters
    ----------
    y : np.ndarray
      The input data to validate

    mono : bool
      Whether or not to force monophonic audio

    Returns
    -------
    valid : bool
        True if all tests pass

    Raises
    ------
    ParameterError
        In any of these cases:
            - `type(y)` is not `np.ndarray`
            - `y.dtype` is not floating-point
            - `mono == True` and `y.ndim` is not 1
            - `mono == False` and `y.ndim` is not 1 or 2
            - `np.isfinite(y).all()` is False
            - `y.flags["F_CONTIGUOUS"]` is False

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> # By default, valid_audio allows only mono signals
    >>> filepath = librosa.util.example_audio_file()
    >>> y_mono, sr = librosa.load(filepath, mono=True)
    >>> y_stereo, _ = librosa.load(filepath, mono=False)
    >>> librosa.util.valid_audio(y_mono), librosa.util.valid_audio(y_stereo)
    True, False

    >>> # To allow stereo signals, set mono=False
    >>> librosa.util.valid_audio(y_stereo, mono=False)
    True

    See also
    --------
    stack
    numpy.asfortranarray
    numpy.float32
    '''

    return True

def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        print (('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)

def frame(x, frame_length=2048, hop_length=512, axis=-1):
    '''Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the input data.

    For example, a one-dimensional input `x = [0, 1, 2, 3, 4, 5, 6]`
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (`axis=-1`), results in the array `x_frames`:

    [[0, 2, 4],
     [1, 3, 5],
     [2, 4, 6]]

    where each column `x_frames[:, i]` contains a contiguous slice of
    the input `x[i * hop_length : i * hop_length + frame_length]`.

    The second way (`axis=0`) results in the array `x_frames`:

    [[0, 1, 2],
     [2, 3, 4],
     [4, 5, 6]]

    where each row `x_frames[i]` contains a contiguous slice of the input.

    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either to the end of the array (`axis=-1`)
    or the beginning of the array (`axis=0`).


    Parameters
    ----------
    x : np.ndarray
        Time series to frame. Must be contiguous in memory, see the "Raises"
        section below for more information.

    frame_length : int > 0 [scalar]
        Length of the frame

    hop_length : int > 0 [scalar]
        Number of steps to advance between frames

    axis : 0 or -1
        The axis along which to frame.

        If `axis=-1` (the default), then `x` is framed along its last dimension.
        `x` must be "F-contiguous" in this case.

        If `axis=0`, then `x` is framed along its first dimension.
        `x` must be "C-contiguous" in this case.

    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES) or (N_FRAMES, frame_length, ...)]
        A framed view of `x`, for example with `axis=-1` (framing on the last dimension):
        `x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]`

        If `axis=0` (framing on the first dimension), then:
        `x_frames[j] = x[j * hop_length : j * hop_length + frame_length]`

    Raises
    ------
    ParameterError
        If `x` is not contiguous in memory or not an `np.ndarray`.

        If `x.shape[axis] < frame_length`, there is not enough data to fill one frame.

        If `hop_length < 1`, frames cannot advance.

        If `axis` is not 0 or -1.  Framing is only supported along the first or last axis.
            If `axis=-1` (the default), then `x` must be "F-contiguous".
            If `axis=0`, then `x` must be "C-contiguous".

        If the contiguity of `x` is incompatible with the framing axis.

    See Also
    --------
    np.asfortranarray : Convert data to F-contiguous representation
    np.ascontiguousarray : Convert data to C-contiguous representation
    np.ndarray.flags : information about the memory layout of a numpy `ndarray`.

    Examples
    --------
    Extract 2048-sample frames from monophonic `y` with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[ 0.000e+00,  0.000e+00, ..., -2.448e-06, -6.789e-07],
           [ 0.000e+00,  0.000e+00, ..., -1.399e-05,  1.004e-06],
           ...,
           [-7.352e-04,  5.162e-03, ...,  0.000e+00,  0.000e+00],
           [ 2.168e-03,  4.870e-03, ...,  0.000e+00,  0.000e+00]],
          dtype=float32)
    >>> y.shape
    (1355168,)
    >>> frames.shape
    (2048, 21143)

    Or frame along the first axis instead of the last:

    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (21143, 2048)

    Frame a stereo signal:

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
    >>> y.shape
    (2, 1355168)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 21143)

    Carve an STFT into fixed-length patches of 32 frames with 50% overlap

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 2647)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 82)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    '''

    if not isinstance(x, np.ndarray):
        print ('Input must be of type numpy.ndarray, '
                             'given type(x)={}'.format(type(x)))

    if x.shape[axis] < frame_length:
        print ('Input is too short (n={:d})'
                             ' for frame_length={:d}'.format(x.shape[axis], frame_length))

    if hop_length < 1:
        print ('Invalid hop_length: {:d}'.format(hop_length))

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        if not x.flags['F_CONTIGUOUS']:
            print ('Input array must be F-contiguous '
                                 'for framing along axis={}'.format(axis))

        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        if not x.flags['C_CONTIGUOUS']:
            print ('Input array must be C-contiguous '
                                 'for framing along axis={}'.format(axis))

        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)
    else:
        print ('Frame axis={} must be either 0 or -1'.format(axis))

    return as_strided(x, shape=shape, strides=strides)

def softmask(X, X_ref, power=1, split_zeros=False):
    '''Robustly compute a softmask operation.

        `M = X**power / (X**power + X_ref**power)`


    Parameters
    ----------
    X : np.ndarray
        The (non-negative) input array corresponding to the positive mask elements

    X_ref : np.ndarray
        The (non-negative) array of reference or background elements.
        Must have the same shape as `X`.

    power : number > 0 or np.inf
        If finite, returns the soft mask computed in a numerically stable way

        If infinite, returns a hard (binary) mask equivalent to `X > X_ref`.
        Note: for hard masks, ties are always broken in favor of `X_ref` (`mask=0`).


    split_zeros : bool
        If `True`, entries where `X` and X`_ref` are both small (close to 0)
        will receive mask values of 0.5.

        Otherwise, the mask is set to 0 for these entries.


    Returns
    -------
    mask : np.ndarray, shape=`X.shape`
        The output mask array

    Raises
    ------
    ParameterError
        If `X` and `X_ref` have different shapes.

        If `X` or `X_ref` are negative anywhere

        If `power <= 0`

    Examples
    --------

    >>> X = 2 * np.ones((3, 3))
    >>> X_ref = np.vander(np.arange(3.0))
    >>> X
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]])
    >>> X_ref
    array([[ 0.,  0.,  1.],
           [ 1.,  1.,  1.],
           [ 4.,  2.,  1.]])
    >>> librosa.util.softmask(X, X_ref, power=1)
    array([[ 1.   ,  1.   ,  0.667],
           [ 0.667,  0.667,  0.667],
           [ 0.333,  0.5  ,  0.667]])
    >>> librosa.util.softmask(X_ref, X, power=1)
    array([[ 0.   ,  0.   ,  0.333],
           [ 0.333,  0.333,  0.333],
           [ 0.667,  0.5  ,  0.333]])
    >>> librosa.util.softmask(X, X_ref, power=2)
    array([[ 1. ,  1. ,  0.8],
           [ 0.8,  0.8,  0.8],
           [ 0.2,  0.5,  0.8]])
    >>> librosa.util.softmask(X, X_ref, power=4)
    array([[ 1.   ,  1.   ,  0.941],
           [ 0.941,  0.941,  0.941],
           [ 0.059,  0.5  ,  0.941]])
    >>> librosa.util.softmask(X, X_ref, power=100)
    array([[  1.000e+00,   1.000e+00,   1.000e+00],
           [  1.000e+00,   1.000e+00,   1.000e+00],
           [  7.889e-31,   5.000e-01,   1.000e+00]])
    >>> librosa.util.softmask(X, X_ref, power=np.inf)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [False, False,  True]], dtype=bool)
    '''
    if X.shape != X_ref.shape:
        print ('Shape mismatch: {}!={}'.format(X.shape,
                                                             X_ref.shape))

    if np.any(X < 0) or np.any(X_ref < 0):
        print ('X and X_ref must be non-negative')

    if power <= 0:
        print ('power must be strictly positive')

    # We're working with ints, cast to float.
    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    # Re-scale the input arrays relative to the larger value
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = (Z < np.finfo(dtype).tiny)
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z)**power
        ref_mask = (X_ref / Z)**power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask

def fix_length(data, size, axis=-1, **kwargs):
    '''Fix the length an array `data` to exactly `size`.

    If `data.shape[axis] < n`, pad according to the provided kwargs.
    By default, `data` is padded with trailing zeros.

    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, 10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, 5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, 10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    Parameters
    ----------
    data : np.ndarray
      array to be length-adjusted

    size : int >= 0 [scalar]
      desired length of the array

    axis : int, <= data.ndim
      axis along which to fix length

    kwargs : additional keyword arguments
        Parameters to `np.pad()`

    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        `data` either trimmed or padded to length `size`
        along the specified axis.

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    '''Normalize an array along a chosen axis.

    Given a norm (described below) and a target axis, the input
    array is scaled so that

        `norm(S, axis=axis) == 1`

    For example, `axis=0` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, `axis=1` normalizes each row of a 2-d array.

    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    `threshold` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.

    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.


    Parameters
    ----------
    S : np.ndarray
        The matrix to normalize

    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : mininum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.

    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least `threshold` are
        normalized.

        By default, the threshold is determined from
        the numerical precision of `S.dtype`.

    fill : None or bool
        If None, then columns (or rows) with norm below `threshold`
        are left as is.

        If False, then columns (rows) with norm below `threshold`
        are set to 0.

        If True, then columns (rows) with norm below `threshold`
        are filled uniformly such that the corresponding norm is 1.

        .. note:: `fill=True` is incompatible with `norm=0` because
            no uniform vector exists with l0 "norm" equal to 1.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array

    Raises
    ------
    ParameterError
        If `norm` is not among the valid types defined above

        If `S` is not finite

        If `fill=True` and `norm=0`

    See Also
    --------
    scipy.linalg.norm

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])

    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])

    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    '''

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        print ('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        print ('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        print ('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            print ('Cannot normalize with norm=0 and fill=True')

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

    elif norm is None:
        return S

    else:
        print ('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm

def tiny(x):
    '''Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in `x`'s
    data type (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is `x.dtype`.

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of `x`.
        If `x` is integer-typed, then the tiny value for `np.float32`
        is returned instead.

    See Also
    --------
    numpy.finfo

    Examples
    --------

    For a standard double-precision floating point number:

    >>> librosa.util.tiny(1.0)
    2.2250738585072014e-308

    Or explicitly as double-precision

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float64))
    2.2250738585072014e-308

    Or complex numbers

    >>> librosa.util.tiny(1j)
    2.2250738585072014e-308

    Single-precision floating point:

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float32))
    1.1754944e-38

    Integer

    >>> librosa.util.tiny(5)
    1.1754944e-38
    '''

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny

def sparsify_rows(x, quantile=0.01):
    '''
    Return a row-sparse matrix approximating the input `x`.

    Parameters
    ----------
    x : np.ndarray [ndim <= 2]
        The input matrix to sparsify.

    quantile : float in [0, 1.0)
        Percentage of magnitude to discard in each row of `x`

    Returns
    -------
    x_sparse : `scipy.sparse.csr_matrix` [shape=x.shape]
        Row-sparsified approximation of `x`

        If `x.ndim == 1`, then `x` is interpreted as a row vector,
        and `x_sparse.shape == (1, len(x))`.

    Raises
    ------
    ParameterError
        If `x.ndim > 2`

        If `quantile` lies outside `[0, 1.0)`

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct a Hann window to sparsify
    >>> x = scipy.signal.hann(32)
    >>> x
    array([ 0.   ,  0.01 ,  0.041,  0.09 ,  0.156,  0.236,  0.326,
            0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
            0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
            0.72 ,  0.625,  0.525,  0.424,  0.326,  0.236,  0.156,
            0.09 ,  0.041,  0.01 ,  0.   ])
    >>> # Discard the bottom percentile
    >>> x_sparse = librosa.util.sparsify_rows(x, quantile=0.01)
    >>> x_sparse
    <1x32 sparse matrix of type '<type 'numpy.float64'>'
        with 26 stored elements in Compressed Sparse Row format>
    >>> x_sparse.todense()
    matrix([[ 0.   ,  0.   ,  0.   ,  0.09 ,  0.156,  0.236,  0.326,
              0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
              0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
              0.72 ,  0.625,  0.525,  0.424,  0.326,  0.236,  0.156,
              0.09 ,  0.   ,  0.   ,  0.   ]])
    >>> # Discard up to the bottom 10th percentile
    >>> x_sparse = librosa.util.sparsify_rows(x, quantile=0.1)
    >>> x_sparse
    <1x32 sparse matrix of type '<type 'numpy.float64'>'
        with 20 stored elements in Compressed Sparse Row format>
    >>> x_sparse.todense()
    matrix([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.326,
              0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
              0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
              0.72 ,  0.625,  0.525,  0.424,  0.326,  0.   ,  0.   ,
              0.   ,  0.   ,  0.   ,  0.   ]])
    '''

    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ParameterError('Input must have 2 or fewer dimensions. '
                             'Provided x.shape={}.'.format(x.shape))

    if not 0.0 <= quantile < 1:
        raise ParameterError('Invalid quantile {:.2f}'.format(quantile))

    x_sparse = lil_matrix(x.shape, dtype=x.dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()

# End Util --------------------------------------------------------------------

# Time Frequencies ------------------------------------------------------------
def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    """Compute the center frequencies of Constant-Q bins.

    Examples
    --------
    >>> # Get the CQT frequencies for 24 notes, starting at C2
    >>> librosa.cqt_frequencies(24, fmin=librosa.note_to_hz('C2'))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
             92.499,   97.999,  103.826,  110.   ,  116.541,  123.471,
            130.813,  138.591,  146.832,  155.563,  164.814,  174.614,
            184.997,  195.998,  207.652,  220.   ,  233.082,  246.942])

    Parameters
    ----------
    n_bins  : int > 0 [scalar]
        Number of constant-Q bins

    fmin    : float > 0 [scalar]
        Minimum frequency

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float
        Deviation from A440 tuning in fractional bins

    Returns
    -------
    frequencies : np.ndarray [shape=(n_bins,)]
        Center frequency for each CQT bin
    """

    correction = 2.0**(float(tuning) / bins_per_octave)
    frequencies = 2.0**(np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies

def note_to_hz(note, **kwargs):
    '''Convert one or more note names to frequency (Hz)

    Examples
    --------
    >>> # Get the frequency of a note
    >>> librosa.note_to_hz('C')
    array([ 16.352])
    >>> # Or multiple notes
    >>> librosa.note_to_hz(['A3', 'A4', 'A5'])
    array([ 220.,  440.,  880.])
    >>> # Or notes with tuning deviations
    >>> librosa.note_to_hz('C2-32', round_midi=False)
    array([ 64.209])

    Parameters
    ----------
    note : str or iterable of str
        One or more note names to convert

    kwargs : additional keyword arguments
        Additional parameters to `note_to_midi`

    Returns
    -------
    frequencies : number or np.ndarray [shape=(len(note),)]
        Array of frequencies (in Hz) corresponding to `note`

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    '''
    return midi_to_hz(note_to_midi(note, **kwargs))

def midi_to_hz(notes):
    """Get the frequency (Hz) of MIDI note(s)

    Examples
    --------
    >>> librosa.midi_to_hz(36)
    65.406

    >>> librosa.midi_to_hz(np.arange(36, 48))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,
             87.307,   92.499,   97.999,  103.826,  110.   ,
            116.541,  123.471])

    Parameters
    ----------
    notes       : int or np.ndarray [shape=(n,), dtype=int]
        midi number(s) of the note(s)

    Returns
    -------
    frequency   : number or np.ndarray [shape=(n,), dtype=float]
        frequency (frequencies) of `notes` in Hz

    See Also
    --------
    hz_to_midi
    note_to_hz
    """

    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0)/12.0))

def note_to_midi(note, round_midi=True):
    '''Convert one or more spelled notes to MIDI number(s).

    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with ``#``, flats may be indicated with ``!`` or ``b``.

    Parameters
    ----------
    note : str or iterable of str
        One or more note names.

    round_midi : bool
        - If `True`, allow for fractional midi notes
        - Otherwise, round cent deviations to the nearest note

    Returns
    -------
    midi : float or np.array
        Midi note numbers corresponding to inputs.

    Raises
    ------
    ParameterError
        If the input is not in valid note format

    See Also
    --------
    midi_to_note
    note_to_hz

    Examples
    --------
    >>> librosa.note_to_midi('C')
    12
    >>> librosa.note_to_midi('C#3')
    49
    >>> librosa.note_to_midi('f4')
    65
    >>> librosa.note_to_midi('Bb-1')
    10
    >>> librosa.note_to_midi('A!8')
    116
    >>> # Lists of notes also work
    >>> librosa.note_to_midi(['C', 'E', 'G'])
    array([12, 16, 19])

    '''

    if not isinstance(note, six.string_types):
        return np.array([note_to_midi(n, round_midi=round_midi) for n in note])

    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1}

    match = re.match(r'^(?P<note>[A-Ga-g])'
                     r'(?P<accidental>[#b!]*)'
                     r'(?P<octave>[+-]?\d+)?'
                     r'(?P<cents>[+-]\d+)?$',
                     note)
    if not match:
        raise ParameterError('Improper note format: {:s}'.format(note))

    pitch = match.group('note').upper()
    offset = np.sum([acc_map[o] for o in match.group('accidental')])
    octave = match.group('octave')
    cents = match.group('cents')

    if not octave:
        octave = 0
    else:
        octave = int(octave)

    if not cents:
        cents = 0
    else:
        cents = int(cents) * 1e-2

    note_value = 12 * (octave + 1) + pitch_map[pitch] + offset + cents

    if round_midi:
        note_value = int(np.round(note_value))

    return note_value

# End Time Frequencies --------------------------------------------------------

# fft -------------------------------------------------------------------------
__FFTLIB = None

def get_n_set_fftlib(lib=None):
    '''Get the FFT library currently used by librosa

    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    '''
    global __FFTLIB
    if lib is None:
        from numpy import fft
        lib = fft

    __FFTLIB = lib

    return __FFTLIB

# End fft ---------------------------------------------------------------------


# Filters ---------------------------------------------------------------------
WINDOW_BANDWIDTHS = {'bart': 1.3334961334912805,
                     'barthann': 1.4560255965133932,
                     'bartlett': 1.3334961334912805,
                     'bkh': 2.0045975283585014,
                     'black': 1.7269681554262326,
                     'blackharr': 2.0045975283585014,
                     'blackman': 1.7269681554262326,
                     'blackmanharris': 2.0045975283585014,
                     'blk': 1.7269681554262326,
                     'bman': 1.7859588613860062,
                     'bmn': 1.7859588613860062,
                     'bohman': 1.7859588613860062,
                     'box': 1.0,
                     'boxcar': 1.0,
                     'brt': 1.3334961334912805,
                     'brthan': 1.4560255965133932,
                     'bth': 1.4560255965133932,
                     'cosine': 1.2337005350199792,
                     'flat': 2.7762255046484143,
                     'flattop': 2.7762255046484143,
                     'flt': 2.7762255046484143,
                     'halfcosine': 1.2337005350199792,
                     'ham': 1.3629455320350348,
                     'hamm': 1.3629455320350348,
                     'hamming': 1.3629455320350348,
                     'han': 1.50018310546875,
                     'hann': 1.50018310546875,
                     'hanning': 1.50018310546875,
                     'nut': 1.9763500280946082,
                     'nutl': 1.9763500280946082,
                     'nuttall': 1.9763500280946082,
                     'ones': 1.0,
                     'par': 1.9174603174603191,
                     'parz': 1.9174603174603191,
                     'parzen': 1.9174603174603191,
                     'rect': 1.0,
                     'rectangular': 1.0,
                     'tri': 1.3331706523555851,
                     'triang': 1.3331706523555851,
                     'triangle': 1.3331706523555851}

def __float_window(window_spec):
    '''Decorator function for windows with fractional input.

    This function guarantees that for fractional `x`, the following hold:

    1. `__float_window(window_function)(x)` has length `np.ceil(x)`
    2. all values from `np.floor(x)` are set to 0.

    For integer-valued `x`, there should be no change in behavior.
    '''

    def _wrap(n, *args, **kwargs):
        '''The wrapped window'''
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))],
                            mode='constant')

        window[n_min:] = 0.0

        return window

    return _wrap

def get_window(window, Nx, fftbins=True):
    '''Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    See Also
    --------
    scipy.signal.get_window

    Notes
    -----
    This function caches at level 10.

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    '''
    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy_get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        print ('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        print ('Invalid window specification: {}'.format(window))

def window_sumsquare(window, n_frames, hop_length=512, win_length=None, n_fft=2048,
                     dtype=np.float32, norm=None):
    '''
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing observations
    in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function

    Examples
    --------
    For a fixed frame length (2048), compare modulation effects for a Hann window
    at different hop lengths:

    >>> n_frames = 50
    >>> wss_256 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=256)
    >>> wss_512 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=512)
    >>> wss_1024 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=1024)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(3,1,1)
    >>> plt.plot(wss_256)
    >>> plt.title('hop_length=256')
    >>> plt.subplot(3,1,2)
    >>> plt.plot(wss_512)
    >>> plt.title('hop_length=512')
    >>> plt.subplot(3,1,3)
    >>> plt.plot(wss_1024)
    >>> plt.title('hop_length=1024')
    >>> plt.tight_layout()
    >>> plt.show()

    '''
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm)**2
    win_sq = pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x

def window_bandwidth(window, n=1000):
    '''Get the equivalent noise bandwidth of a window function.


    Parameters
    ----------
    window : callable or string
        A window function, or the name of a window function.
        Examples:
        - scipy.signal.hann
        - 'boxcar'

    n : int > 0
        The number of coefficients to use in estimating the
        window bandwidth

    Returns
    -------
    bandwidth : float
        The equivalent noise bandwidth (in FFT bins) of the
        given window function

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    get_window
    '''

    if hasattr(window, '__name__'):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        win = get_window(window, n)
        WINDOW_BANDWIDTHS[key] = n * np.sum(win**2) / np.sum(np.abs(win))**2

    return WINDOW_BANDWIDTHS[key]

def constant_q(sr, fmin=None, n_bins=84, bins_per_octave=12, tuning=Deprecated(),
               window='hann', filter_scale=1, pad_fft=True, norm=1,
               dtype=np.complex64, **kwargs):
    r'''Construct a constant-Q basis.

    This uses the filter bank described by [1]_.

    .. [1] McVicar, Matthew.
            "A machine learning approach to automatic chord extraction."
            Dissertation, University of Bristol. 2013.


    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float [scalar] <DEPRECATED>
        Tuning deviation from A440 in fractions of a bin

        .. note:: This parameter is deprecated in 0.7.1.  It will be removed in
                  version 0.8.

    window : string, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the `mode=` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 64-bit (single precision) complex floating point.

    kwargs : additional keyword arguments
        Arguments to `np.pad()` when `pad==True`.

    Returns
    -------
    filters : np.ndarray, `len(filters) == n_bins`
        `filters[i]` is `i`\ th time-domain CQT basis filter

    lengths : np.ndarray, `len(lengths) == n_bins`
        The (fractional) length of each filter

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    constant_q_lengths
    librosa.core.cqt
    librosa.util.normalize


    Examples
    --------
    Use a shorter window for each filter

    >>> basis, lengths = librosa.filters.constant_q(22050, filter_scale=0.5)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.constant_q(22050)
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(2, 1, 1)
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     plt.plot(i + f_scale.real)
    ...     plt.plot(i + f_scale.imag, linestyle=':')
    >>> plt.axis('tight')
    >>> plt.yticks(np.arange(len(notes[:12])), notes[:12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filters (one octave, time domain)')
    >>> plt.xlabel('Time (samples at 22050 Hz)')
    >>> plt.legend(['Real', 'Imaginary'], frameon=True, framealpha=0.8)
    >>> plt.subplot(2, 1, 2)
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear')
    >>> plt.yticks(np.arange(len(notes))[::12], notes[::12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filter magnitudes (frequency domain)')
    >>> plt.tight_layout()
    >>> plt.show()
    '''

    if fmin is None:
        fmin = note_to_hz('C1')

    if isinstance(tuning, Deprecated):
        tuning = 0.0
    else:
        warnings.warn('The `tuning` parameter to `filters.constant_q` '
                      'is deprecated in librosa 0.7.1. '
                      'It will be removed in 0.8.0.', DeprecationWarning)

    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)
    fmin = correction * fmin

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(sr, fmin,
                                 n_bins=n_bins,
                                 bins_per_octave=bins_per_octave,
                                 window=window,
                                 filter_scale=filter_scale)

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Convert lengths back to frequencies
    freqs = Q * sr / lengths

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(np.arange(-ilen//2, ilen//2, dtype=float) * 1j * 2 * np.pi * freq / sr)

        # Apply the windowing function
        sig = sig * __float_window(window)(len(sig))

        # Normalize
        sig = normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0**(np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray([pad_center(filt, max_len, **kwargs)
                          for filt in filters], dtype=dtype)

    return filters, np.asarray(lengths)


def constant_q_lengths(sr, fmin, n_bins=84, bins_per_octave=12,
                       tuning=Deprecated(), window='hann', filter_scale=1):
    r'''Return length of each filter in a constant-Q basis.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin.

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float [scalar] <DEPRECATED>
        Tuning deviation from A440 in fractions of a bin

        .. note:: This parameter is deprecated in 0.7.1.  It will be removed in
                  version 0.8.

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    constant_q
    librosa.core.cqt
    '''

    if fmin <= 0:
        raise ParameterError('fmin must be positive')

    if bins_per_octave <= 0:
        raise ParameterError('bins_per_octave must be positive')

    if filter_scale <= 0:
        raise ParameterError('filter_scale must be positive')

    if n_bins <= 0 or not isinstance(n_bins, int):
        raise ParameterError('n_bins must be a positive integer')

    if isinstance(tuning, Deprecated):
        tuning = 0.0
    else:
        warnings.warn('The `tuning` parameter to `filters.constant_q_lengths` is deprecated in librosa 0.7.1.'
                      'It will be removed in 0.8.0.', DeprecationWarning)

    correction = 2.0**(float(tuning) / bins_per_octave)
    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Compute the frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    if freq[-1] * (1 + 0.5 * window_bandwidth(window) / Q) > sr / 2.0:
        raise ParameterError('Filter pass-band lies beyond Nyquist')

    # Convert frequencies to filter lengths
    lengths = Q * sr / freq

    return lengths


# End Filters -----------------------------------------------------------------


# Spectrum --------------------------------------------------------------------
def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    """Short-time Fourier transform (STFT). [1]_ (chapter 2)

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - `np.abs(D[f, t])` is the magnitude of frequency bin `f`
      at frame `t`, and

    - `np.angle(D[f, t])` is the phase of frequency bin `f`
      at frame `t`.

    The integers `t` and `f` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.

    .. [1] M. Müller. "Fundamentals of Music Processing." Springer, 2015


    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        input signal

    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix `D` is (1 + n_fft/2).
        The default value, n_fft=2048 samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting `n_fft` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in `D` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to `win_length / 4` (see below).

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()` of length `win_length`
        and then padded with zeros to match `n_fft`.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization tradeoff and needs to be adjusted
        according to the properties of the input signal `y`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:

        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`

        - a window function, such as `scipy.signal.hanning`

        - a vector or array of length `n_fft`


        Defaults to a raised cosine window ("hann"), which is adequate for
        most applications in audio signal processing.

        .. see also:: `filters.get_window`

    center : boolean
        If `True`, the signal `y` is padded so that frame
        `D[:, t]` is centered at `y[t * hop_length]`.

        If `False`, then `D[:, t]` begins at `y[t * hop_length]`.

        Defaults to `True`,  which simplifies the alignment of `D` onto a
        time grid by means of `librosa.core.frames_to_samples`.
        Note, however, that `center` must be set to `False` when analyzing
        signals with `librosa.stream`.

        .. see also:: `stream`

    dtype : numeric type
        Complex numeric type for `D`.  Default is single-precision
        floating-point complex (`np.complex64`).

    pad_mode : string or function
        If `center=True`, this argument is passed to `np.pad` for padding
        the edges of the signal `y`. By default (`pad_mode="reflect"`),
        `y` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If `center=False`,  this argument is ignored.

        .. see also:: `np.pad`


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.


    See Also
    --------
    istft : Inverse STFT

    reassigned_spectrogram : Time-frequency reassigned spectrogram


    Notes
    -----
    This function caches at level 20.


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = np.abs(librosa.stft(y))
    >>> D
    array([[2.58028018e-03, 4.32422794e-02, 6.61255598e-01, ...,
            6.82710262e-04, 2.51654536e-04, 7.23036574e-05],
           [2.49403086e-03, 5.15930466e-02, 6.00107312e-01, ...,
            3.48026224e-04, 2.35853557e-04, 7.54836728e-05],
           [7.82410789e-04, 1.05394892e-01, 4.37517226e-01, ...,
            6.29352580e-04, 3.38571583e-04, 8.38094638e-05],
           ...,
           [9.48568513e-08, 4.74725084e-07, 1.50052492e-05, ...,
            1.85637656e-08, 2.89708542e-08, 5.74304337e-09],
           [1.25165826e-07, 8.58259284e-07, 1.11157215e-05, ...,
            3.49099771e-08, 3.11740926e-08, 5.29926236e-09],
           [1.70630571e-07, 8.92518756e-07, 1.23656537e-05, ...,
            5.33256745e-08, 3.33264900e-08, 5.13272980e-09]], dtype=float32)

    Use left-aligned frames, instead of centered frames

    >>> D_left = np.abs(librosa.stft(y, center=False))


    Use a shorter hop length

    >>> D_short = np.abs(librosa.stft(y, hop_length=64))


    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.amplitude_to_db(D,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    >>> plt.show()
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # fft = get_n_set_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = np.fft.rfft(fft_window *
                                             y_frames[:, bl_s:bl_t],
                                             axis=0)
    return stft_matrix

def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample:(sample + n_fft)] += ytmp[:, frame]

def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, dtype=np.float32, length=None):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

        .. see also:: `filters.get_window`

    center : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    length : int > 0, optional
        If provided, the output `y` is zero-padded or clipped to exactly
        `length` samples.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of `y` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    1.4901161e-07
    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = pad_center(ifft_window, n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(
            stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = int(MAX_MEM_BLOCK // (stft_matrix.shape[0] *
                                           stft_matrix.itemsize))

    # fft = get_n_set_fftlib()

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * np.fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length:], ytmp, hop_length)

        frame += (bl_t - bl_s)

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(window,
                                       n_frames,
                                       win_length=win_length,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       dtype=dtype)

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2):-int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = fix_length(y[start:], length)

    return y

def magphase(D, power=1):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.


    Returns
    -------
    D_mag : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`, raised to `power`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[  2.524e-03,   4.329e-02, ...,   3.217e-04,   3.520e-05],
           [  2.645e-03,   5.152e-02, ...,   3.283e-04,   3.432e-04],
           ...,
           [  1.966e-05,   9.828e-06, ...,   3.164e-07,   9.370e-06],
           [  1.966e-05,   9.830e-06, ...,   3.161e-07,   9.366e-06]], dtype=float32)
    >>> phase
    array([[  1.000e+00 +0.000e+00j,   1.000e+00 +0.000e+00j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j],
           [  1.000e+00 +1.615e-16j,   9.950e-01 -1.001e-01j, ...,
              9.794e-01 +2.017e-01j,   1.492e-02 -9.999e-01j],
           ...,
           [  1.000e+00 -5.609e-15j,  -5.081e-04 +1.000e+00j, ...,
             -9.549e-01 -2.970e-01j,   2.938e-01 -9.559e-01j],
           [ -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j]], dtype=complex64)


    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[  0.000e+00,   0.000e+00, ...,   3.142e+00,   3.142e+00],
           [  1.615e-16,  -1.003e-01, ...,   2.031e-01,  -1.556e+00],
           ...,
           [ -5.609e-15,   1.571e+00, ...,  -2.840e+00,  -1.273e+00],
           [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]], dtype=float32)

    """

    mag = np.abs(D)
    mag **= power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

# End Spectrum ----------------------------------------------------------------


# Decompose -------------------------------------------------------------------
def hpss(S, kernel_size=31, power=2.0, mask=False, margin=1.0):
    """Median-filtering harmonic percussive source separation (HPSS).

    If `margin = 1.0`, decomposes an input spectrogram `S = H + P`
    where `H` contains the harmonic components,
    and `P` contains the percussive components.

    If `margin > 1.0`, decomposes an input spectrogram `S = H + P + R`
    where `R` contains residual components not included in `H` or `P`.

    This implementation is based upon the algorithm described by [1]_ and [2]_.

    .. [1] Fitzgerald, Derry.
        "Harmonic/percussive separation using median filtering."
        13th International Conference on Digital Audio Effects (DAFX10),
        Graz, Austria, 2010.

    .. [2] Driedger, Müller, Disch.
        "Extending harmonic-percussive separation of audio."
        15th International Society for Music Information Retrieval Conference (ISMIR 2014),
        Taipei, Taiwan, 2014.

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        input spectrogram. May be real (magnitude) or complex.

    kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
        kernel size(s) for the median filters.

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the width of the
          harmonic filter, and the second value specifies the width
          of the percussive filter.

    power : float > 0 [scalar]
        Exponent for the Wiener filter when constructing soft mask matrices.

    mask : bool
        Return the masking matrices instead of components.

        Masking matrices contain non-negative real values that
        can be used to measure the assignment of energy from `S`
        into harmonic or percussive components.

        Components can be recovered by multiplying `S * mask_H`
        or `S * mask_P`.


    margin : float or tuple (margin_harmonic, margin_percussive)
        margin size(s) for the masks (as described in [2]_)

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the margin of the
          harmonic mask, and the second value specifies the margin
          of the percussive mask.

    Returns
    -------
    harmonic : np.ndarray [shape=(d, n)]
        harmonic component (or mask)

    percussive : np.ndarray [shape=(d, n)]
        percussive component (or mask)


    See Also
    --------
    util.softmask

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Separate into harmonic and percussive

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> D = librosa.stft(y)
    >>> H, P = librosa.decompose.hpss(D)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Full power spectrogram')
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(H),
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Harmonic power spectrogram')
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(P),
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Percussive power spectrogram')
    >>> plt.tight_layout()
    >>> plt.show()


    Or with a narrower horizontal filter

    >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

    Just get harmonic/percussive masks, not the spectra

    >>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
    >>> mask_H
    array([[  1.000e+00,   1.469e-01, ...,   2.648e-03,   2.164e-03],
           [  1.000e+00,   2.368e-01, ...,   9.413e-03,   7.703e-03],
           ...,
           [  8.869e-01,   5.673e-02, ...,   4.603e-02,   1.247e-05],
           [  7.068e-01,   2.194e-02, ...,   4.453e-02,   1.205e-05]], dtype=float32)
    >>> mask_P
    array([[  2.858e-05,   8.531e-01, ...,   9.974e-01,   9.978e-01],
           [  1.586e-05,   7.632e-01, ...,   9.906e-01,   9.923e-01],
           ...,
           [  1.131e-01,   9.433e-01, ...,   9.540e-01,   1.000e+00],
           [  2.932e-01,   9.781e-01, ...,   9.555e-01,   1.000e+00]], dtype=float32)

    Separate into harmonic/percussive/residual components by using a margin > 1.0

    >>> H, P = librosa.decompose.hpss(D, margin=3.0)
    >>> R = D - (H+P)
    >>> y_harm = librosa.core.istft(H)
    >>> y_perc = librosa.core.istft(P)
    >>> y_resi = librosa.core.istft(R)


    Get a more isolated percussive component by widening its margin

    >>> H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))

    """

    if np.iscomplexobj(S):
        S, phase = magphase(S)
    else:
        phase = 1

    if np.isscalar(kernel_size):
        win_harm = kernel_size
        win_perc = kernel_size
    else:
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]

    if np.isscalar(margin):
        margin_harm = margin
        margin_perc = margin
    else:
        margin_harm = margin[0]
        margin_perc = margin[1]

    # margin minimum is 1.0
    if margin_harm < 1 or margin_perc < 1:
        print ("Margins must be >= 1.0. "
                             "A typical range is between 1 and 10.")

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

    split_zeros = (margin_harm == 1 and margin_perc == 1)

    mask_harm = softmask(harm, perc * margin_harm,
                              power=power,
                              split_zeros=split_zeros)

    mask_perc = softmask(perc, harm * margin_perc,
                              power=power,
                              split_zeros=split_zeros)

    if mask:
        return mask_harm, mask_perc

    return ((S * mask_harm) * phase, (S * mask_perc) * phase)

# End Decompose ---------------------------------------------------------------


# Audio -----------------------------------------------------------------------
BW_FASTEST = resampy.filters.get_filter('kaiser_fast')[2]

def __audioread_load(path, offset, duration, dtype):
    '''Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    '''

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native

def resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):
    """Resample a time series from orig_sr to target_sr

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.

    orig_sr : number > 0 [scalar]
        original sampling rate of `y`

    target_sr : number > 0 [scalar]
        target sampling rate

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            To use a faster method, set `res_type='kaiser_fast'`.

            To use `scipy.signal.resample`, set `res_type='fft'` or `res_type='scipy'`.

            To use `scipy.signal.resample_poly`, set `res_type='polyphase'`.

        .. note::
            When using `res_type='polyphase'`, only integer sampling rates are
            supported.

    fix : bool
        adjust the length of the resampled signal to be of size exactly
        `ceil(target_sr * len(y) / orig_sr)`

    scale : bool
        Scale the resampled signal so that `y` and `y_hat` have approximately
        equal total energy.

    kwargs : additional keyword arguments
        If `fix==True`, additional keyword arguments to pass to
        `librosa.util.fix_length`.

    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        `y` resampled from `orig_sr` to `target_sr`

    Raises
    ------
    ParameterError
        If `res_type='polyphase'` and `orig_sr` or `target_sr` are not both
        integer-valued.

    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy.resample

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Downsample from 22 KHz to 8 KHz

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((1355168,), (491671,))
    """

    # First, validate the audio buffer
    valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    # if res_type in ('scipy', 'fft'):
    #     y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    # elif res_type == 'polyphase':
    #     if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
    #         print ('polyphase resampling is only supported for integer-valued sampling rates.')

    #     # For polyphase resampling, we need up- and down-sampling ratios
    #     # We can get those from the greatest common divisor of the rates
    #     # as long as the rates are integrable
    #     orig_sr = int(orig_sr)
    #     target_sr = int(target_sr)
    #     gcd = np.gcd(orig_sr, target_sr)
    #     y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    # else:
    y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)

# End Audio -------------------------------------------------------------------

# Constant Q ------------------------------------------------------------------
def __cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
                     filter_scale, norm, sparsity, hop_length=None,
                     window='hann'):
    '''Generate the frequency domain constant-Q filter basis.'''

    basis, lengths = constant_q(sr,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        filter_scale=filter_scale,
                                        norm=norm,
                                        pad_fft=True,
                                        window=window)

    # Filters are padded up to the nearest integral power of 2
    
    n_fft = basis.shape[1]

    if (hop_length is not None and
            n_fft < 2.0**(1 + np.ceil(np.log2(hop_length)))):

        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

    # sparsify the basis
    fft_basis = sparsify_rows(fft_basis, quantile=sparsity)

    return fft_basis, n_fft, lengths

def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos

def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    '''Compute the number of early downsampling operations'''

    downsample_count1 = max(0, int(np.ceil(np.log2(BW_FASTEST * nyquist /
                                                   filter_cutoff)) - 1) - 1)

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)

def __early_downsample(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff, scale):
    '''Perform early downsampling on an audio signal, if it applies.'''

    downsample_count = __early_downsample_count(nyquist, filter_cutoff,
                                                hop_length, n_octaves)

    if downsample_count > 0 and res_type == 'kaiser_fast':
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor

        if len(y) < downsample_factor:
            raise ParameterError('Input signal length={:d} is too short for '
                                 '{:d}-octave CQT'.format(len(y), n_octaves))

        new_sr = sr / float(downsample_factor)
        y = resample(y, sr, new_sr,
                           res_type=res_type,
                           scale=True)

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length

def __cqt_response(y, n_fft, hop_length, fft_basis, mode):
    '''Compute the filter response with a target STFT hop.'''

    # Compute the STFT matrix
    D = stft(y, n_fft=n_fft, hop_length=hop_length,
             window=1,
             pad_mode=mode)

    # And filter response energy
    return fft_basis.dot(D)

# End Constant Q  -------------------------------------------------------------

# Top Level Functions ---------------------------------------------------------
def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type='kaiser_best'):
    """Load an audio file as a floating point time series.

    Audio will be automatically resampled to the given rate
    (default `sr=22050`).

    To preserve the native sampling rate of the file, use `sr=None`.

    Parameters
    ----------
    path : string, int, pathlib.Path or file-like object
        path to the input file.

        Any codec supported by `soundfile` or `audioread` will work.

        Any string file paths, or any object implementing Python's
        file interface (e.g. `pathlib.Path`) are supported as `path`.

        If the codec is supported by `soundfile`, then `path` can also be
        an open file descriptor (int).

    sr   : number > 0 [scalar]
        target sampling rate

        'None' uses the native sampling rate

    mono : bool
        convert signal to mono

    offset : float
        start reading after this time (in seconds)

    duration : float
        only load up to this much audio (in seconds)

    dtype : numeric type
        data type of `y`

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            For alternative resampling modes, see `resample`

        .. note::
           `audioread` may truncate the precision of the audio data to 16 bits.

           See https://librosa.github.io/librosa/ioformats.html for alternate
           loading methods.


    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series

    sr   : number > 0 [scalar]
        sampling rate of `y`


    Examples
    --------
    >>> # Load an ogg vorbis file
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename)
    >>> y
    array([ -4.756e-06,  -6.020e-06, ...,  -1.040e-06,   0.000e+00], dtype=float32)
    >>> sr
    22050

    >>> # Load a file and resample to 11 KHz
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([ -2.077e-06,  -2.928e-06, ...,  -4.395e-06,   0.000e+00], dtype=float32)
    >>> sr
    11025

    >>> # Load 5 seconds of a file, starting 15 seconds in
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([ 0.069,  0.1  , ..., -0.101,  0.   ], dtype=float32)
    >>> sr
    22050

    """
    print('load')
    y, sr_native = __audioread_load(path, offset, duration, dtype)

    # Final cleanup for dtype and contiguity

    if sr is not None:
        y = resample(y, sr_native, sr, res_type=res_type)

    else:
        sr = sr_native

    return y, sr

def get_duration(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                 center=True, filename=None):
    """Compute the duration (in seconds) of an audio time series,
    feature matrix, or filename.

    Examples
    --------
    >>> # Load the example audio file
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.get_duration(y=y, sr=sr)
    61.45886621315193

    >>> # Or directly from an audio file
    >>> librosa.get_duration(filename=librosa.util.example_audio_file())
    61.4

    >>> # Or compute duration from an STFT matrix
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = librosa.stft(y)
    >>> librosa.get_duration(S=S, sr=sr)
    61.44

    >>> # Or a non-centered STFT matrix
    >>> S_left = librosa.stft(y, center=False)
    >>> librosa.get_duration(S=S_left, sr=sr)
    61.3471201814059

    Parameters
    ----------
    y : np.ndarray [shape=(n,), (2, n)] or None
        audio time series

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        STFT matrix, or any STFT-derived matrix (e.g., chromagram
        or mel spectrogram).
        Durations calculated from spectrogram inputs are only accurate
        up to the frame resolution. If high precision is required,
        it is better to use the audio time series directly.

    n_fft       : int > 0 [scalar]
        FFT window size for `S`

    hop_length  : int > 0 [ scalar]
        number of audio samples between columns of `S`

    center  : boolean
        - If `True`, `S[:, t]` is centered at `y[t * hop_length]`
        - If `False`, then `S[:, t]` begins at `y[t * hop_length]`

    filename : str
        If provided, all other parameters are ignored, and the
        duration is calculated directly from the audio file.
        Note that this avoids loading the contents into memory,
        and is therefore useful for querying the duration of
        long files.

        As in `load()`, this can also be an integer or open file-handle
        that can be processed by `soundfile`.

    Returns
    -------
    d : float >= 0
        Duration (in seconds) of the input time series or spectrogram.

    Raises
    ------
    ParameterError
        if none of `y`, `S`, or `filename` are provided.

    Notes
    -----
    `get_duration` can be applied to a file (`filename`), a spectrogram (`S`),
    or audio buffer (`y, sr`).  Only one of these three options should be
    provided.  If you do provide multiple options (e.g., `filename` and `S`),
    then `filename` takes precedence over `S`, and `S` takes precedence over
    `(y, sr)`.
    """
    print('get_duration')

    if filename is not None:
        try:
            return sf.info(filename).duration
        except RuntimeError:
            with audioread.audio_open(filename) as fdesc:
                return fdesc.duration

    if y is None:
        if S is None:
            print ('At least one of (y, sr), S, or filename must be provided')

        n_frames = S.shape[1]
        n_samples = n_fft + hop_length * (n_frames - 1)

        # If centered, we lose half a window from each end of S
        if center:
            n_samples = n_samples - 2 * int(n_fft / 2)

    else:
        # Ensure Fortran contiguity.
        y = np.asfortranarray(y)

        # Validate the audio buffer.  Stereo is okay here.
        valid_audio(y, mono=False)
        if y.ndim == 1:
            n_samples = len(y)
        else:
            n_samples = y.shape[-1]

    return float(n_samples) / sr

def hpss(y, **kwargs):
    '''Decompose an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform `y`.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series
    kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.


    Returns
    -------
    y_harmonic : np.ndarray [shape=(n,)]
        audio time series of the harmonic elements

    y_percussive : np.ndarray [shape=(n,)]
        audio time series of the percussive elements

    See Also
    --------
    harmonic : Extract only the harmonic component
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS on spectrograms


    Examples
    --------
    >>> # Extract harmonic and percussive components
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y)

    >>> # Get a more isolated percussive component by widening its margin
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))

    '''
    print('hpss')

    # Compute the STFT matrix
    h_stft = stft(y)

    # Decompose into harmonic and percussives
    stft_harm, stft_perc = hpss(h_stft, **kwargs)

    # Invert the STFTs.  Adjust length to match the input.
    y_harm = fix_length(istft(stft_harm, dtype=y.dtype), len(y))
    y_perc = fix_length(istft(stft_perc, dtype=y.dtype), len(y))

    return y_harm, y_perc

def cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, tuning=0.0, filter_scale=1,
        norm=1, sparsity=0.01, window='hann',
        scale=True, pad_mode='reflect', res_type=None):
    '''Compute the constant-Q transform of an audio signal.

    This implementation is based on the recursive sub-sampling method
    described by [1]_.

    .. [1] Schoerkhuber, Christian, and Anssi Klapuri.
        "Constant-Q transform toolbox for music processing."
        7th Sound and Music Computing Conference, Barcelona, Spain. 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to C1 ~= 32.70 Hz

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at `fmin`

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If `None`, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting CQT will be modified to
        `fmin * 2**(tuning / bins_per_octave)`.

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If `True`, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to `norm='ortho'` in FFT.

        If `False`, do not scale the CQT. This is analogous to
        `norm=None` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.core.stft` and `np.pad`.

    res_type : string [optional]
        The resampling mode for recursive downsampling.

        By default, `cqt` will adaptively select a resampling mode
        which trades off accuracy at high frequencies for efficiency at low frequencies.

        You can override this by specifying a resampling mode as supported by
        `librosa.core.resample`.  For example, `res_type='fft'` will use a high-quality,
        but potentially slow FFT-based down-sampling, while `res_type='polyphase'` will
        use a fast, but potentially inaccurate down-sampling.

    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.complex or np.float]
        Constant-Q value each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length` is not an integer multiple of
        `2**(n_bins / bins_per_octave)`

        Or if `y` is too short to support the frequency range of the CQT.

    See Also
    --------
    librosa.core.resample
    librosa.util.normalize

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Generate and plot a constant-Q power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = np.abs(librosa.cqt(y, sr=sr))
    >>> librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                          sr=sr, x_axis='time', y_axis='cqt_note')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Constant-Q power spectrum')
    >>> plt.tight_layout()
    >>> plt.show()


    Limit the frequency range

    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60))
    >>> C
    array([[  8.827e-04,   9.293e-04, ...,   3.133e-07,   2.942e-07],
           [  1.076e-03,   1.068e-03, ...,   1.153e-06,   1.148e-06],
           ...,
           [  1.042e-07,   4.087e-07, ...,   1.612e-07,   1.928e-07],
           [  2.363e-07,   5.329e-07, ...,   1.294e-07,   1.611e-07]])


    Using a higher frequency resolution

    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60 * 2, bins_per_octave=12 * 2))
    >>> C
    array([[  1.536e-05,   5.848e-05, ...,   3.241e-07,   2.453e-07],
           [  1.856e-03,   1.854e-03, ...,   2.397e-08,   3.549e-08],
           ...,
           [  2.034e-07,   4.245e-07, ...,   6.213e-08,   1.463e-07],
           [  4.896e-08,   5.407e-07, ...,   9.176e-08,   1.051e-07]])
    '''
    print('cqt')
    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    len_orig = len(y)

    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

    # Apply tuning correction
    fmin = fmin * 2.0**(tuning / bins_per_octave)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth(window) / Q)
    nyquist = sr / 2.0

    auto_resample = False
    if not res_type:
        auto_resample = True
        if filter_cutoff < BW_FASTEST * nyquist:
            res_type = 'kaiser_fast'
        else:
            res_type = 'kaiser_best'

    y, sr, hop_length = __early_downsample(y, sr, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff, scale)

    cqt_resp = []

    if auto_resample and res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               filter_scale,
                                               norm,
                                               sparsity,
                                               window=window)

        # Compute the CQT filter response and append it to the stack
        cqt_resp.append(__cqt_response(y, n_fft, hop_length, fft_basis, pad_mode))

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth(window) / Q)

        res_type = 'kaiser_fast'

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        print ('hop_length must be a positive integer '
                             'multiple of 2^{0:d} for {1:d}-octave CQT'
                             .format(n_octaves - 1, n_octaves))

    # Now do the recursive bit
    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                           n_filters,
                                           bins_per_octave,
                                           filter_scale,
                                           norm,
                                           sparsity,
                                           window=window)

    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        # Resample (except first time)
        if i > 0:
            if len(my_y) < 2:
                print ('Input signal length={} is too short for '
                                     '{:d}-octave CQT'.format(len_orig,
                                                              n_octaves))

            my_y = resample(my_y, 2, 1,
                                  res_type=res_type,
                                  scale=True)
            # The re-scale the filters to compensate for downsampling
            fft_basis[:] *= np.sqrt(2)

            my_sr /= 2.0
            my_hop //= 2

        # Compute the cqt filter response and append to the stack
        cqt_resp.append(__cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode))

    C = __trim_stack(cqt_resp, n_bins)

    if scale:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             window=window,
                                             filter_scale=filter_scale)
        C /= np.sqrt(lengths[:, np.newaxis])

    return C

# End Top Level Functions -----------------------------------------------------