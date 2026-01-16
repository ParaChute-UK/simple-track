import sys

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from src.OpticalFlowSolver import OpticalFlowSolver

import numpy as np
import pytest

of_solver = OpticalFlowSolver()
mwe_domain = np.zeros((100, 100), dtype=int)

# If subdomains are chosen properly, we don't expect there to be much difference between
# the advection of features across the subdomain. Therefore, we don't test large differences
# between dy, dx values estimated from large discrepancies in similar sizes feature displacements.
# However, we do expect more weight to be given to larger features so this is tested here.


def run_of_solver_test(field0, field1, expected_dy, expected_dx):
    dy, dx = of_solver.track_subdomain_flow(field0, field1)
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
