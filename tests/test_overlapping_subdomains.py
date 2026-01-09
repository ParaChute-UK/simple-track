import sys

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from src.OpticalFlowSolver import OpticalFlowSolver
import numpy as np


def test_overlapping_subdomain_bounds_with_perfect_fit():
    """
    Test that the overlapping subdomain bounds are calculated correctly.
    """

    # Create an OpticalFlowSolver instance with a squarelength that does not
    # divide the feature field shape exactly
    of_solver = OpticalFlowSolver()

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

    # Create an OpticalFlowSolver instance with a squarelength that does not
    # divide the feature field shape exactly
    of_solver = OpticalFlowSolver()

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

    # Create an OpticalFlowSolver instance with a squarelength that does not
    # divide the feature field shape exactly
    of_solver = OpticalFlowSolver()

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


def test_check_subdomain_size_fits_in_full_domain_with_odd_shape():
    """
    Test that the check_subdomain_size_fits_in_full_domain returns
    False if an odd subdomain dimension is provided
    """
    of_solver = OpticalFlowSolver()
    feature_field_shape = (12, 16)
    sd_shape = (3, 4)

    result = of_solver.check_subdomain_size_fits_in_full_domain(
        feature_field_shape, sd_shape
    )
    if result:
        raise Exception("Test should have returned False")


def check_subdomain_size_fits_in_full_domain_with_even_incorrect_shape():
    """
    Test that the check_subdomain_size_fits_in_full_domain returns
    False if an odd subdomain dimension is provided
    """
    of_solver = OpticalFlowSolver()
    feature_field_shape = (12, 16)
    sd_shape = (8, 10)

    result = of_solver.check_subdomain_size_fits_in_full_domain(
        feature_field_shape, sd_shape
    )
    if result:
        raise Exception("Test should have returned False")


def check_subdomain_size_fits_in_full_domain_with_even_correct_shape():
    """
    Test that the check_subdomain_size_fits_in_full_domain returns
    False if an odd subdomain dimension is provided
    """
    of_solver = OpticalFlowSolver()
    feature_field_shape = (12, 16)
    sd_shape = (3, 4)

    result = of_solver.check_subdomain_size_fits_in_full_domain(
        feature_field_shape, sd_shape
    )
    if not result:
        raise Exception("Test should have returned True")
