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
