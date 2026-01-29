import sys

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from flow_solver import OpticalFlowSolver

import numpy as np

of_solver = OpticalFlowSolver()


def test_get_2d_tukey_window_with_equal_dims():
    """
    Test the get_2d_tukey_window with small dimensions and equal shape
    """
    arr_shape = (5, 5)
    tukey_window = of_solver.get_2d_tukey_window(arr_shape)

    expected_window = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0.25, 0.5, 0.25, 0],
            [0, 0.5, 1, 0.5, 0],
            [0, 0.25, 0.5, 0.25, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_allclose(tukey_window, expected_window)


def test_get_2d_tukey_window_with_unequal_dims():
    """
    Test the get_2d_tukey_window with small dimensions and unequal shape
    """
    arr_shape = (7, 5)
    tukey_window = of_solver.get_2d_tukey_window(arr_shape)

    expected_window = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.125, 0.25, 0.125, 0.0],
        [0.0, 0.375, 0.75, 0.375, 0.0],
        [0.0, 0.5, 1.0, 0.5, 0.0],
        [0.0, 0.375, 0.75, 0.375, 0.0],
        [0.0, 0.125, 0.25, 0.125, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    np.testing.assert_allclose(tukey_window, expected_window)


def test_get_2d_tukey_window_with_float_in_input():
    """
    Test the get_2d_tukey_window returns TypeError with float input
    """
    arr_shape = (5.5, 5)
    try:
        of_solver.get_2d_tukey_window(arr_shape)
    except TypeError:
        pass


def test_get_2d_tukey_window_with_int_input():
    """
    Test the get_2d_tukey_window returns TypeError with scalar input
    """
    arr_shape = 5
    try:
        of_solver.get_2d_tukey_window(arr_shape)
    except TypeError:
        pass


def test_get_2d_tukey_window_with_3d_shape():
    """
    Test the get_2d_tukey_window returns ValueError with 3d input
    """
    arr_shape = (5, 4, 7)
    try:
        of_solver.get_2d_tukey_window(arr_shape)
    except ValueError:
        pass
