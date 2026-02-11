import numpy as np


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
            raise TypeError("args must be an array-like (array, list or tuple)")

    # Check each array has the required shape
    if shape is not None:
        for arr in modified_args:
            if arr.shape != shape:
                msg = f"Argument with shape {arr.shape} does not have required shape {shape}"
                raise ValueError(msg)

    # Check each array has required number of dimensions
    if ndim is not None:
        for arr in modified_args:
            if arr.ndim != ndim:
                msg = (
                    f"Argument with ndim {arr.ndim} does not have required ndim {ndim}"
                )
                raise ValueError(msg)

    # Check each array has the required dtype
    if dtype is not None:
        # Change python base types to numpy types for looser comparison
        if dtype is int:
            dtype = np.integer
        if dtype is float:
            dtype = np.floating

        for arr in modified_args:
            if not np.issubdtype(arr.dtype, dtype):
                msg = f"Argument with dtype {arr.dtype} does not have required dtype {dtype}"
                raise TypeError(msg)

    # Check each input array is equal size
    if equal_shape:
        arr0_shape = args[0].shape
        if not all([arr.shape == arr0_shape for arr in modified_args]):
            msg = f"Input array shapes differ: {[arr.shape for arr in args]}"
            raise ValueError(msg)

    # Check all values are positive
    if non_negative:
        if not all([np.all(arr >= 0) for arr in modified_args]):
            msg = "Expected inputs to contain non-negative values"
            raise ValueError(msg)

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
