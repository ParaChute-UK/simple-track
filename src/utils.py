import numpy as np


def check_arrays(*args, shape=None, ndim=None, dtype=None, equal_shape=False):
    # Check inputs args are array like, convert to numpy array if possible,
    # otherwise return TypeError
    modified_args = []
    for arr in args:
        if isinstance(arr, np.ndarray):
            modified_args.append(arr)
        elif isinstance(arr, (list, tuple)):
            modified_args.append(np.array(arr))
        else:
            raise TypeError("args must be an array, list or tuple")

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
        for arr in modified_args:
            if not all(np.issubdtype(i, dtype) for i in arr):
                msg = f"Argument with dtype {arr.dtype} does not have required dtype {dtype}"
                raise TypeError(msg)

    # Check each input array is equal size
    if equal_shape:
        arr0_shape = args[0].shape
        if not all([arr.shape == arr0_shape for arr in args]):
            msg = f"Input array shapes differ: {[arr.shape for arr in args]}"
            raise ValueError(msg)

    return modified_args
