import numpy as np
import pytest

from simpletrack.utils import (
    ArrayShapeError,
    ArrayTypeError,
    FloatIDError,
    IDError,
    NegativeIDError,
    ZeroIDError,
    check_arrays,
    check_valid_ids,
)


def test_check_arrays_with_ndarray_inputs():
    """
    Test that check_arrays returns unmodified NDArrays
    """
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(20)

    new_arr1, new_arr2 = check_arrays(test_arr1, test_arr2)
    np.testing.assert_array_equal(test_arr1, new_arr1)
    np.testing.assert_array_equal(test_arr2, new_arr2)


def test_check_arrays_with_list_tuple_inputs():
    """
    Test that check_arrays returns NDArrays from lists and tuples
    """
    test_list = [2, 3, 4]
    test_tuple = (5, 6, 7)

    new_arr1, new_arr2 = check_arrays(test_list, test_tuple)
    if not all(isinstance(arr, np.ndarray) for arr in [new_arr1, new_arr2]):
        raise TypeError("List/tuple not converted to NDArray as expected")

    np.testing.assert_array_equal(new_arr1, np.array(test_list))
    np.testing.assert_array_equal(new_arr2, np.array(test_tuple))


def test_check_arrays_with_equal_shapes():
    """
    Test that check_arrays does not throw an error when checking equal shapes
    """
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    check_arrays(test_arr1, test_arr2, equal_shape=True)


def test_check_arrays_with_unequal_shapes():
    """
    Test that check_arrays does not throw an error when checking equal shapes
    """
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 21)
    try:
        check_arrays(test_arr1, test_arr2, equal_shape=True)
    except ArrayShapeError:
        pass


def test_check_arrays_against_correct_shape():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_shape = (10,)
    check_arrays(test_arr1, test_arr2, shape=expected_shape)


def test_check_arrays_against_incorrect_shape():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_shape = (20,)
    try:
        check_arrays(test_arr1, test_arr2, shape=expected_shape)
    except ArrayShapeError:
        pass


def test_check_arrays_against_correct_ndim():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_ndim = 1
    check_arrays(test_arr1, test_arr2, ndim=expected_ndim)


def test_check_arrays_against_incorrect_ndim():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_ndim = 2
    try:
        check_arrays(test_arr1, test_arr2, ndim=expected_ndim)
    except ArrayShapeError:
        pass


def test_check_arrays_against_correct_int_dtype():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_dtype = int
    check_arrays(test_arr1, test_arr2, dtype=expected_dtype)


def test_check_arrays_against_incorrect_float_dtype():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_dtype = float
    try:
        check_arrays(test_arr1, test_arr2, dtype=expected_dtype)
    except ArrayTypeError:
        pass


def test_check_arrays_against_incorrect_str_dtype():
    test_arr1 = np.arange(10)
    test_arr2 = np.arange(10, 20)
    expected_dtype = str
    try:
        check_arrays(test_arr1, test_arr2, dtype=expected_dtype)
    except ArrayTypeError:
        pass


def test_check_arrays_with_non_negative_values():
    test_arr1 = np.arange(-5, 5)
    try:
        check_arrays(test_arr1, non_negative=True)
    except ArrayTypeError:
        pass


def test_check_arrays_with_no_non_negative_values():
    test_arr1 = np.arange(0, 10)
    check_arrays(test_arr1, non_negative=True)


def test_check_valid_ids_valid_inputs():
    test_id1 = 13
    test_id2 = list(range(5))
    test_id3 = np.arange(9)
    # Should accept values that can safely be converted to int
    # This is checked in the code by doing val == int(val), which in python
    # will equate to True if the numeric values are the same
    test_id4 = 4.0
    test_id5 = np.array((5.0, 6.0))
    results = check_valid_ids(test_id1, test_id2, test_id3, test_id4, test_id5)
    # Check that the output ids have been converted to int
    assert type(results[3]) is int
    assert np.issubdtype(results[4].dtype, np.integer)


@pytest.mark.parametrize(
    "input, expected_error",
    [
        [1.4, FloatIDError],
        [-4, NegativeIDError],
        [0, ZeroIDError],
        [np.arange(-5, 5, dtype=int), NegativeIDError],
        [np.arange(1.5, 10.5, 1, dtype=float), FloatIDError],
        ["not an int", IDError],
    ],
)
def test_check_valid_ids_invalid_inputs(input, expected_error):
    try:
        check_valid_ids(input)
    except expected_error:
        pass
