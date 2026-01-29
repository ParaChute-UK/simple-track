import sys
import pytest

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from flow_solver import OpticalFlowSolver, pairwise_with_stride
import numpy as np

of_solver = OpticalFlowSolver()


@pytest.mark.parametrize(
    "iterable, stride, expected_output",
    [
        [(0, 10, 20, 30), 1, ((0, 10), (10, 20), (20, 30))],
        [(0, 10, 20, 30), 2, ((0, 20), (10, 30))],
        [(0, 10, 20, 30), 3, ((0, 30),)],
        [(0, 10, 20, 30), 4, ()],
    ],
)
def test_pairwise_with_stride(iterable, stride, expected_output):
    pairwise_iter = pairwise_with_stride(iterable, stride)
    assert tuple(pairwise_iter) == expected_output


def test_subdomain_iter_valid_inputs():
    y_subdomain_bounds = (0, 10, 20, 30)
    x_subdomain_bounds = (0, 15, 30, 45)
    expected_iter = (
        ((0, 20), (0, 30)),
        ((0, 20), (15, 45)),
        ((10, 30), (0, 30)),
        ((10, 30), (15, 45)),
    )

    subdom_iter = of_solver.subdomain_iter(y_subdomain_bounds, x_subdomain_bounds)
    assert expected_iter == tuple(subdom_iter)


@pytest.mark.parametrize(
    "domain_shape, subdomain_shape, expected",
    [
        [(100, 100), (20, 20), (9, 9)],
        [(80, 80), (10, 10), (15, 15)],
        [(80, 100), (20, 20), (7, 9)],
        [(100, 100), (20, 10), (9, 19)],
        [(100, -100), (10, 10), ValueError],
        [(100.5, 100), (10, 10), TypeError],
        [(100, 100, 100), (10, 10), ValueError],
    ],
)
def test_get_subdomain_containment_arrays(domain_shape, subdomain_shape, expected):
    try:
        test_arr, __ = of_solver.get_subdomain_containment_arrays(
            domain_shape, subdomain_shape
        )
        assert test_arr.shape == expected
    except expected:
        pass


def test_overlapping_subdomain_bounds_with_perfect_fit():
    """
    Test that the overlapping subdomain bounds are calculated correctly.
    """

    # Create a dummy feature field of shape (12, 16)
    feature_field_shape = (12, 16)

    # get subdomain shape from squarelength
    squarelength = 4
    subdomain_shape = np.array((squarelength, squarelength), dtype=int)

    # Get the subdomain bounds
    subdomain_y_bounds, subdomain_x_bounds = of_solver.get_overlapping_subdomain_idxs(
        feature_field_shape=feature_field_shape, subdomain_shape=subdomain_shape
    )

    # Expected subdomain bounds for squarelength of 4 and feature field shape of (10, 10)
    expected_y_bounds = np.array((0, 2, 4, 6, 8, 10, 12))
    expected_x_bounds = np.array((0, 2, 4, 6, 8, 10, 12, 14, 16))

    np.testing.assert_array_equal(
        subdomain_y_bounds, expected_y_bounds, "y bounds do not match expected values."
    )
    np.testing.assert_array_equal(
        subdomain_x_bounds, expected_x_bounds, "x bounds do not match expected values."
    )


def test_overlapping_subdomain_bounds_with_odd_subdomain():
    """
    Test that the overlapping subdomain bounds returns error with odd-shaped subdomain.
    """

    # Create a dummy feature field of shape (12, 16)
    feature_field_shape = (12, 16)

    # get subdomain shape from squarelength
    squarelength = 3
    subdomain_shape = np.array((squarelength, squarelength), dtype=int)

    # Get the subdomain bounds
    try:
        subdomain_y_bounds, subdomain_x_bounds = (
            of_solver.get_overlapping_subdomain_idxs(
                feature_field_shape=feature_field_shape, subdomain_shape=subdomain_shape
            )
        )
    except ValueError:
        pass


def test_overlapping_subdomain_bounds_without_fit():
    """
    Test that the overlapping subdomain bounds return error if the requested shape
    does not fit exactly within the full domain.
    """

    # Create a dummy feature field of shape (12, 16)
    feature_field_shape = (12, 16)

    # get subdomain shape from squarelength
    squarelength = 6
    subdomain_shape = np.array((squarelength, squarelength), dtype=int)

    # Get the subdomain bounds
    try:
        subdomain_y_bounds, subdomain_x_bounds = (
            of_solver.get_overlapping_subdomain_idxs(
                feature_field_shape=feature_field_shape, subdomain_shape=subdomain_shape
            )
        )
    except ValueError:
        pass


@pytest.mark.parametrize(
    "feature_field_shape, subdomain_shape, expected_result",
    [
        [(100, 100), (10, 10), True],
        [(50, 50), (10, 10), True],
        [(100, 100), (25, 25), False],  # Cannot be odd, overlap is not at a grid point
        [(100, 100), (30, 30), False],  # Does not fit in domain
        [(-10, -10), (2, 2), False],  # Does not accept negative values
        [(100, 100), (5.5, 5.5), TypeError],  # Does not accept floats
        [("abc", "abc"), (5, 5), TypeError],  # Does not accept strings
        [(100, 100, 100), (50, 50, 50), False],  # Does not accept 3d fields
        [(100,), (50,), False],  # Does not accept 1D fields
    ],
)
def test_check_sufficient_subdomain_size(
    feature_field_shape, subdomain_shape, expected_result
):
    try:
        test_result = of_solver.check_subdomain_size_fits_in_full_domain(
            feature_field_shape, subdomain_shape
        )
        assert test_result == expected_result, (
            f"Expected {expected_result}, got {test_result}"
        )
    except expected_result:
        pass


def test_check_subdomain_variability_unchanged():
    subdomain_vals = np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    of_solver = OpticalFlowSolver(subdomain_tolerance=3)
    filtered_vals = of_solver.check_subdomain_variability(subdomain_vals)
    np.testing.assert_array_equal(filtered_vals, subdomain_vals)


def test_check_subdomain_variability_single_outlier():
    subdomain_vals = np.array([[5.0, 6.0, 20.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    of_solver = OpticalFlowSolver(subdomain_tolerance=3)
    filtered_vals = of_solver.check_subdomain_variability(subdomain_vals)
    expected_vals = np.array([[5.0, 6.0, np.nan], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    np.testing.assert_array_equal(filtered_vals, expected_vals)


def test_check_subdomain_variability_mutiple_outliers():
    subdomain_vals = np.array([[20, 6.0, 7.0], [5.0, 6.0, 20], [5.0, 6.0, 7.0]])
    of_solver = OpticalFlowSolver(subdomain_tolerance=3)
    filtered_vals = of_solver.check_subdomain_variability(subdomain_vals)
    expected_vals = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [5.0, 6.0, np.nan]]
    )
    np.testing.assert_array_equal(filtered_vals, expected_vals)


def test_check_subdomain_variability_mutiple_neighbouring_outliers():
    subdomain_vals = np.array([[5.0, 23, 23], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    of_solver = OpticalFlowSolver(subdomain_tolerance=3)
    filtered_vals = of_solver.check_subdomain_variability(subdomain_vals)
    expected_vals = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [5.0, 6.0, 7.0]]
    )
    np.testing.assert_array_equal(filtered_vals, expected_vals)


def test_check_subdomain_variability_mutiple_neighbouring_outliers_higher_tolerance():
    subdomain_vals = np.array([[5.0, 23, 23], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    of_solver = OpticalFlowSolver(subdomain_tolerance=10)
    filtered_vals = of_solver.check_subdomain_variability(subdomain_vals)
    np.testing.assert_array_equal(filtered_vals, subdomain_vals)
