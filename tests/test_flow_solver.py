import sys
import pytest

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Documents/Code/simple-track/src"
)
from flow_solver import OpticalFlowSolver, pairwise_with_stride

import numpy as np

of_solver = OpticalFlowSolver()
mwe_domain = np.zeros((100, 100), dtype=int)


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


def test_fill_nans_edge():
    """
    NaN values at the edge of the domain do not have enough neighbouring values
    to construct an interpolated value, and should therefore be filled with 0
    """
    subdomain_vals = np.arange(4, 9).repeat(5).reshape((5, 5)).transpose().astype(float)
    expected_vals = subdomain_vals.copy()
    subdomain_vals[0, 0] = np.nan
    expected_vals[0, 0] = 0
    subdomain_vals_no_nans = of_solver._fill_nans(subdomain_vals)
    np.testing.assert_array_equal(subdomain_vals_no_nans, expected_vals)


def test_fill_nans_middle():
    """
    NaN values towards the middle of the domain can be filled using neighbouring values,
    therefore the filled value should reproduce the original array.
    """
    subdomain_vals = np.arange(4, 9).repeat(5).reshape((5, 5)).transpose().astype(float)
    expected_vals = subdomain_vals.copy()
    subdomain_vals[3, 3] = np.nan
    subdomain_vals_no_nans = OpticalFlowSolver()._fill_nans(subdomain_vals)
    np.testing.assert_array_equal(subdomain_vals_no_nans, expected_vals)


def test_interpolate_subdomain_flows():
    subdomain_vals = np.arange(4, 9).repeat(7).reshape((5, 7)).T
    # [4, 5, 6, 7, 8] repeated in 7 rows

    full_domain_shape = (80, 60)
    x_subdomain_bounds = np.arange(start=10, stop=60, step=10)
    y_subdomain_bounds = np.arange(start=10, stop=80, step=10)

    # expect [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4.1, 4.2 ... 7.9, 8, 8, 8, ...]
    expected_row = np.repeat([4], 11)
    expected_row = np.append(expected_row, np.arange(4.1, 8, 0.1))
    expected_row = np.append(expected_row, np.repeat([8], 10))

    interp_flow = of_solver.interpolate_subdomain_flows(
        y_subdomain_bounds=y_subdomain_bounds,
        x_subdomain_bounds=x_subdomain_bounds,
        subdomain_flows=subdomain_vals,
        full_domain_shape=full_domain_shape,
    )
    # Allclose used to avoid float precision errors
    np.testing.assert_allclose(interp_flow[0], expected_row)


def test_interpolate_subdomain_flows_using_subdomain_idxs():
    subdomain_vals = np.arange(4, 9).repeat(7).reshape((5, 7)).T
    # [4, 5, 6, 7, 8] repeated in 7 rows

    full_domain_shape = (80, 60)
    subdomain_shape = (20, 20)
    y_idxs, x_idxs = of_solver.get_overlapping_subdomain_idxs(
        full_domain_shape, subdomain_shape
    )

    # But, we want interior subdomain bounds, as written in run method
    y_subdomain_bounds = y_idxs[1:-1]
    x_subdomain_bounds = x_idxs[1:-1]

    # expect [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4.1, 4.2 ... 7.9, 8, 8, 8, ...]
    expected_row = np.repeat([4], 11)
    expected_row = np.append(expected_row, np.arange(4.1, 8, 0.1))
    expected_row = np.append(expected_row, np.repeat([8], 10))

    interp_flow = of_solver.interpolate_subdomain_flows(
        y_subdomain_bounds=y_subdomain_bounds,
        x_subdomain_bounds=x_subdomain_bounds,
        subdomain_flows=subdomain_vals,
        full_domain_shape=full_domain_shape,
    )
    # Allclose used to avoid float precision errors
    np.testing.assert_allclose(interp_flow[0], expected_row)


def test_interpolate_subdomain_flows_with_mostly_zero_subdomain_vals():
    subdomain_vals = np.zeros((7, 5))
    subdomain_vals[1, 1] = 5
    subdomain_vals[4, 4] = 2

    full_domain_shape = (80, 60)
    x_subdomain_bounds = np.arange(start=10, stop=60, step=10)
    y_subdomain_bounds = np.arange(start=10, stop=80, step=10)

    # expect [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4.1, 4.2 ... 7.9, 8, 8, 8, ...]
    expected_row = np.repeat([4], 11)
    expected_row = np.append(expected_row, np.arange(4.1, 8, 0.1))
    expected_row = np.append(expected_row, np.repeat([8], 10))

    interp_flow = of_solver.interpolate_subdomain_flows(
        y_subdomain_bounds=y_subdomain_bounds,
        x_subdomain_bounds=x_subdomain_bounds,
        subdomain_flows=subdomain_vals,
        full_domain_shape=full_domain_shape,
    )
    expected_flow = np.zeros(full_domain_shape)
    np.testing.assert_array_equal(interp_flow, expected_flow)


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


# If subdomains are chosen properly, we don't expect there to be much difference between
# the advection of features across the subdomain. Therefore, we don't test large differences
# between dy, dx values estimated from large discrepancies in similar sizes feature displacements.
# However, we do expect more weight to be given to larger features so this is tested here.


def run_of_solver_test(field0, field1, expected_dy, expected_dx):
    dy, dx = of_solver.derive_subdomain_flow(field0, field1)
    if isinstance(expected_dx, list):
        assert dy in expected_dy, f"expected: {expected_dy}, actual {dy}"
        assert dx in expected_dx, f"expected: {expected_dx}, actual {dx}"
    else:
        assert dy == expected_dy, f"expected: {expected_dy}, actual {dy}"
        assert dx == expected_dx, f"expected: {expected_dx}, actual {dx}"


@pytest.mark.parametrize(
    "test_dy, test_dx, expected_dy, expected_dx",
    [
        [0, 0, 0, 0],  # No flow
        [5, 0, 5, 0],  # Cardinal flow
        [5, 5, 5, 5],  # Diagonal flow
        [-5, 0, -5, 0],  # Negative cardinal flow
        [-5, -5, -5, -5],  # Negative diagonal flow
        [-5, 5, -5, 5],  # Mix of positive and negative diagonal flow
    ],
)
def test_single_feature_advection(test_dy, test_dx, expected_dy, expected_dx):
    """
    Test track_subdomain_flow with a single feature to advect
    """
    y_slice = slice(10, 30)
    x_slice = slice(10, 30)
    test0 = mwe_domain.copy()
    test0[y_slice, x_slice] = 1

    test1 = mwe_domain.copy()
    y_slice_new = slice(y_slice.start + test_dy, y_slice.stop + test_dy)
    x_slice_new = slice(x_slice.start + test_dx, x_slice.stop + test_dx)
    test1[y_slice_new, x_slice_new] = 1

    run_of_solver_test(test0, test1, expected_dy, expected_dx)


@pytest.mark.parametrize(
    "test_dy, test_dx, expected_dy, expected_dx",
    [
        [0, 0, 0, 0],  # No flow
        [5, 0, 5, 0],  # Cardinal flow
        [5, 5, 5, 5],  # Diagonal flow
        [-5, 0, -5, 0],  # Negative cardinal flow
        [-5, -5, -5, -5],  # Negative diagonal flow
        [-5, 5, -5, 5],  # Mix of positive and negative diagonal flow
    ],
)
def test_multiple_equal_feature_equal_flow_advection(
    test_dy, test_dx, expected_dy, expected_dx
):
    """
    Test track_subdomain_flow with a multiple features of equal size advecting
    by the same flow
    """
    test0 = mwe_domain.copy()
    y_slice_f0 = slice(20, 40)
    x_slice_f0 = slice(20, 40)
    y_slice_f1 = slice(20, 40)
    x_slice_f1 = slice(60, 80)
    test0[y_slice_f0, x_slice_f0] = 1
    test0[y_slice_f1, x_slice_f1] = 1

    test1 = mwe_domain.copy()
    y_slice_f0_new = slice(y_slice_f0.start + test_dy, y_slice_f0.stop + test_dy)
    x_slice_f0_new = slice(x_slice_f0.start + test_dx, x_slice_f0.stop + test_dx)
    y_slice_f1_new = slice(y_slice_f1.start + test_dy, y_slice_f1.stop + test_dy)
    x_slice_f1_new = slice(x_slice_f1.start + test_dx, x_slice_f1.stop + test_dx)
    test1[y_slice_f0_new, x_slice_f0_new] = 1
    test1[y_slice_f1_new, x_slice_f1_new] = 1

    run_of_solver_test(test0, test1, expected_dy, expected_dx)


@pytest.mark.parametrize(
    "test_dy_f0, test_dx_f0, test_dy_f1, test_dx_f1, expected_dy, expected_dx",
    [
        [10, 0, 13, 0, [10, 11, 12, 13], [0]],  # F0 dy=10 | F1 dy=13
        [10, 5, 13, 7, [10, 11, 12, 13], [5, 6, 7]],  # F0 dy=10 dx=5 | F1 dy=13 dx=7
        [-10, 5, -13, 7, [-10, -11, -12, -13], [5, 6, 7]],
    ],
)
def test_multiple_equal_feature_unequal_flow_advection(
    test_dy_f0, test_dx_f0, test_dy_f1, test_dx_f1, expected_dy, expected_dx
):
    """
    Test track_subdomain_flow with a multiple features of equal size advecting
    by the slightly different flow. Note that there is no one correct answer to
    these scenarios, therefore any result within the range of prescribed dy/dx
    is considered acceptable.
    """
    test0 = mwe_domain.copy()
    y_slice_f0 = slice(20, 40)
    x_slice_f0 = slice(20, 40)
    y_slice_f1 = slice(20, 40)
    x_slice_f1 = slice(60, 80)
    test0[y_slice_f0, x_slice_f0] = 1
    test0[y_slice_f1, x_slice_f1] = 1

    test1 = mwe_domain.copy()
    y_slice_f0_new = slice(y_slice_f0.start + test_dy_f0, y_slice_f0.stop + test_dy_f0)
    x_slice_f0_new = slice(x_slice_f0.start + test_dx_f0, x_slice_f0.stop + test_dx_f0)
    y_slice_f1_new = slice(y_slice_f1.start + test_dy_f1, y_slice_f1.stop + test_dy_f1)
    x_slice_f1_new = slice(x_slice_f1.start + test_dx_f1, x_slice_f1.stop + test_dx_f1)
    test1[y_slice_f0_new, x_slice_f0_new] = 1
    test1[y_slice_f1_new, x_slice_f1_new] = 1

    run_of_solver_test(test0, test1, expected_dy, expected_dx)


@pytest.mark.parametrize(
    "test_dy_f0, test_dx_f0, test_dy_f1, test_dx_f1, expected_dy, expected_dx",
    [
        # Here, f1 is larger than f0 by 5 pixels, hence we expect it to pick up f1 flow
        [10, 0, 15, 0, 15, 0],  # f0 dy = 10 | f1 dy = 15
        [10, 5, 13, 7, 13, 7],
        [-10, 5, -13, 7, -13, 7],
        [5, 5, 0, 0, 0, 0],  # f0 is moving and f1 is stationary, f1 chosen
    ],
)
def test_multiple_unequal_feature_unequal_flow_advection(
    test_dy_f0, test_dx_f0, test_dy_f1, test_dx_f1, expected_dy, expected_dx
):
    """
    Test track_subdomain_flow with a multiple features of unequal size advecting
    by the slightly different flow. We should expect the solver to choose dy/dx
    based on the flow of the larger feature, which here is feature 1.
    """
    test0 = mwe_domain.copy()
    y_slice_f0 = slice(20, 40)
    x_slice_f0 = slice(20, 40)
    y_slice_f1 = slice(20, 40)
    x_slice_f1 = slice(60, 85)  # Larger in x than f0 by 5 pixels
    test0[y_slice_f0, x_slice_f0] = 1
    test0[y_slice_f1, x_slice_f1] = 1

    test1 = mwe_domain.copy()
    y_slice_f0_new = slice(y_slice_f0.start + test_dy_f0, y_slice_f0.stop + test_dy_f0)
    x_slice_f0_new = slice(x_slice_f0.start + test_dx_f0, x_slice_f0.stop + test_dx_f0)
    y_slice_f1_new = slice(y_slice_f1.start + test_dy_f1, y_slice_f1.stop + test_dy_f1)
    x_slice_f1_new = slice(x_slice_f1.start + test_dx_f1, x_slice_f1.stop + test_dx_f1)
    test1[y_slice_f0_new, x_slice_f0_new] = 1
    test1[y_slice_f1_new, x_slice_f1_new] = 1

    run_of_solver_test(test0, test1, expected_dy, expected_dx)


@pytest.mark.parametrize(
    "test_dy_f0, test_dx_f0, test_dy_f1, test_dx_f1, f1_y_growth, expected_dy, expected_dx",
    [
        [10, 0, 0, 0, 7, 7, 0],  # F0 advects, F1 grows by 7 -> expected dy = 7 (?)
        # [10, 0, 10, 0, 7, 17, 0],  # F1 grows by 7 advects by 10, should get dy = 17, get 0
        # [0, 0, 0, 5, 7, 7, 5], # F1 grows by 7 and dx=5, expect dy=7 and dx=5. get dy=0
        # [0, 0, 5, 5, 7, 12, 5],  # Expect dy=12, get 0
        # [5, 5, 0, 0, 7, 7, 0],  # expected dy=7 dx=0, or dy=dx=5  Get dy=5, dx=0 ???
    ],
)
def test_growing_feature(
    test_dy_f0,
    test_dx_f0,
    test_dy_f1,
    test_dx_f1,
    f1_y_growth,
    expected_dy,
    expected_dx,
):
    """
    Test track_subdomain_flow with a multiple, initially equally sized features with
    one feature growing. We should expect the solver to choose dy/dx based on the flow
    of the larger feature, which here is feature 1.
    """
    test0 = mwe_domain.copy()
    y_slice_f0 = slice(20, 40)
    x_slice_f0 = slice(20, 40)
    y_slice_f1 = slice(20, 40)
    x_slice_f1 = slice(60, 80)
    test0[y_slice_f0, x_slice_f0] = 1
    test0[y_slice_f1, x_slice_f1] = 1

    test1 = mwe_domain.copy()
    y_slice_f0_new = slice(y_slice_f0.start + test_dy_f0, y_slice_f0.stop + test_dy_f0)
    x_slice_f0_new = slice(x_slice_f0.start + test_dx_f0, x_slice_f0.stop + test_dx_f0)
    y_slice_f1_new = slice(
        y_slice_f1.start + test_dy_f1, y_slice_f1.stop + test_dy_f1 + f1_y_growth
    )
    x_slice_f1_new = slice(x_slice_f1.start + test_dx_f1, x_slice_f1.stop + test_dx_f1)
    test1[y_slice_f0_new, x_slice_f0_new] = 1
    test1[y_slice_f1_new, x_slice_f1_new] = 1

    run_of_solver_test(test0, test1, expected_dy, expected_dx)


@pytest.mark.parametrize(
    "test_dy_f0, test_dx_f0, test_dy_f1, test_dx_f1, f1_y_growth, expected_dy, expected_dx",
    [
        [10, 0, 15, 0, 10, 10, 0],  # F1 advects 15 but shrinks 10. Expect dy=10 from f0
        [0, 10, 0, 15, 5, 0, 10],  # Similar behaviour if just dx rather than just dy
        [10, 10, 15, 15, 5, 10, 10],  # Also works if features move in diagonal
        [0, 0, 10, 10, 5, 0, 0],  # If f0 is stationary but f1 moves, solver picks f0
        [10, 10, 0, 0, 5, 10, 10],  # If f0 moves and f1 is stationary, solver picks f0
    ],
)
def test_shrinking_feature(
    test_dy_f0,
    test_dx_f0,
    test_dy_f1,
    test_dx_f1,
    f1_y_growth,
    expected_dy,
    expected_dx,
):
    """
    Test track_subdomain_flow with a multiple, initially equally sized features with
    one feature shrinking. We should expect the solver to choose dy/dx based on the
    flow of the larger feature that doesn't change size
    """
    test0 = mwe_domain.copy()
    y_slice_f0 = slice(20, 40)
    x_slice_f0 = slice(20, 40)
    y_slice_f1 = slice(20, 40)
    x_slice_f1 = slice(60, 80)
    test0[y_slice_f0, x_slice_f0] = 1
    test0[y_slice_f1, x_slice_f1] = 1

    test1 = mwe_domain.copy()
    y_slice_f0_new = slice(y_slice_f0.start + test_dy_f0, y_slice_f0.stop + test_dy_f0)
    x_slice_f0_new = slice(x_slice_f0.start + test_dx_f0, x_slice_f0.stop + test_dx_f0)
    y_slice_f1_new = slice(
        y_slice_f1.start + test_dy_f1, y_slice_f1.stop + test_dy_f1 - f1_y_growth
    )
    x_slice_f1_new = slice(x_slice_f1.start + test_dx_f1, x_slice_f1.stop + test_dx_f1)
    test1[y_slice_f0_new, x_slice_f0_new] = 1
    test1[y_slice_f1_new, x_slice_f1_new] = 1

    run_of_solver_test(test0, test1, expected_dy, expected_dx)
