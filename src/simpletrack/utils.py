import numpy as np

from simpletrack.exceptions import (
    ArrayShapeError,
    ArrayTypeError,
    FloatIDError,
    IDError,
    NegativeIDError,
    ZeroIDError,
)


def check_arrays(
    *args, shape=None, ndim=None, dtype=None, equal_shape=False, non_negative=False
):
    # Check inputs args are array like, convert to numpy array if possible,
    # otherwise return TypeError
    modified_args = []
    for arr in args:
        if isinstance(arr, np.ndarray):
            modified_args.append(arr)
        elif isinstance(arr, (list, tuple)):
            modified_args.append(np.array(arr))
        else:
            raise ArrayTypeError("args must be an array-like (array, list or tuple)")

    # Check each array has the required shape
    if shape is not None:
        for arr in modified_args:
            if arr.shape != shape:
                msg = f"""
                Argument with shape {arr.shape} does not have required shape {shape}
                """
                raise ArrayShapeError(msg)

    # Check each array has required number of dimensions
    if ndim is not None:
        for arr in modified_args:
            if arr.ndim != ndim:
                msg = (
                    f"Argument with ndim {arr.ndim} does not have required ndim {ndim}"
                )
                raise ArrayShapeError(msg)

    # Check each array has the required dtype
    if dtype is not None:
        # Change python base types to numpy types for looser comparison
        if dtype is int:
            np_dtype = np.integer
        elif dtype is float:
            np_dtype = np.floating
        else:
            raise ArrayTypeError(f"Unsupported dtype {dtype} for check_arrays")

        for arr in modified_args:
            if not np.issubdtype(arr.dtype, np_dtype):
                try:
                    arr = arr.astype(dtype, casting="same_value")
                except (ValueError, TypeError):
                    msg = f"""
                    Argument with dtype {arr.dtype} does not have and cannot be cast to
                    required dtype {dtype}
                    """
                    raise ArrayTypeError(msg) from None

    # Check each input array is equal size
    if equal_shape:
        arr0_shape = args[0].shape
        if not all([arr.shape == arr0_shape for arr in modified_args]):
            msg = f"Input array shapes differ: {[arr.shape for arr in args]}"
            raise ArrayShapeError(msg)

    # Check all values are positive
    if non_negative:
        if not all([np.all(arr >= 0) for arr in modified_args]):
            msg = "Expected inputs to contain non-negative values"
            raise ArrayTypeError(msg)

    # Don't want to return a single arg input as a list
    if len(modified_args) == 1:
        return modified_args[0]
    else:
        return modified_args


def check_valid_ids(*args):
    """
    Checks that all inputs (scalar or vector) contain valid id data - each element
    is a positive, nonzero integer
    """
    modified_args = []
    for arg in args:
        if isinstance(arg, str):
            raise IDError("Cannot interpret str as ID")
        elif np.isscalar(arg):
            arg_native = native(arg)
            # Check if turning input into int would not change its value
            # If so, continue checks with int version
            if int(arg_native) == arg_native:
                arg_native = int(arg_native)
            if not np.issubdtype(type(arg_native), np.integer):
                raise FloatIDError(f"{arg_native} not an int")
            if arg_native == 0:
                raise ZeroIDError("Valid IDs start from 1, got 0")
            if arg_native < 0:
                raise NegativeIDError(f"Valid IDs start from 1, got {arg_native}")
            modified_args.append(arg_native)

        else:  # Looking at vector inputs
            arg_array = np.array(arg) if isinstance(arg, (list, tuple)) else arg
            if len(arg_array) == 0:
                return []

            # Check if turning input into int would not change its value
            # If so, continue checks with int version
            if np.all(arg_array.astype(int) == arg_array):
                arg_array = arg_array.astype(int)

            if not np.issubdtype(arg_array.dtype, np.integer):
                raise FloatIDError(f"Array must contain ints only: {arg_array}")
            if any(arg_array < 0):
                raise NegativeIDError(
                    f"Array must contain positive ints only: {arg_array}"
                )
            modified_args.append(arg_array)

    # Don't want to return a single arg input as a list
    if len(modified_args) == 1:
        return modified_args[0]
    else:
        return modified_args


def native(value):
    """
    Convert numpy scalar types to native python types.
    If argument is already native, return unchanged

    Args:
        value (any): Input value

    Returns:
        any: Converted value
    """
    return getattr(value, "tolist", lambda: value)()
