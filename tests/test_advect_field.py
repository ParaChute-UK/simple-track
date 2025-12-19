import numpy as np
import sys

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from src.FrameTracker import advect_field_using_motion_vectors


def test_advect_field():
    """
    Test the ability of the advect_field_using_motion_vectors function to
    advect a simple feature field
    """
    test_field = np.zeros((10, 10), dtype=int)
    test_y_flow = np.zeros_like(test_field)
    test_x_flow = np.zeros_like(test_field)

    feature_mask_y = slice(3, 6)
    feature_mask_x = slice(2, 4)
    test_field[feature_mask_y, feature_mask_x] = 1
    test_y_flow[feature_mask_y, feature_mask_x] = 1
    test_x_flow[feature_mask_y, feature_mask_x] = 1

    expected_field = np.zeros_like(test_field)
    expected_field[4:7, 3:5] = 1

    advected_field = advect_field_using_motion_vectors(
        test_field, test_y_flow, test_x_flow
    )

    err_msg = "Test failed: advected array is not equal to expected array."
    np.testing.assert_array_equal(advected_field, expected_field, err_msg)


def test_advect_field_with_feature_conflict():
    """
    Test the advect_field_using_motion_vectors functtion to choose the correct feature
    during an advection conflict. Feature should be he one where the centroid
    is closest to the conflict area.
    """

    test_field = np.zeros((10, 10), dtype=int)
    test_y_flow = np.zeros_like(test_field)
    test_x_flow = np.zeros_like(test_field)

    feature1_mask_y = slice(3, 6)
    feature1_mask_x = slice(2, 4)
    test_field[feature1_mask_y, feature1_mask_x] = 1
    test_y_flow[feature1_mask_y, feature1_mask_x] = 1
    test_x_flow[feature1_mask_y, feature1_mask_x] = 1

    # Have feature 2 also move into the same area as feature 1, but it is
    # more elongated in x direction so its centroid should be further
    # from the conflicting areas
    feature2_mask_y = slice(3, 6)
    feature2_mask_x = slice(5, 9)
    test_field[feature2_mask_y, feature2_mask_x] = 2
    test_y_flow[feature2_mask_y, feature2_mask_x] = 1
    test_x_flow[feature2_mask_y, feature2_mask_x] = -1
    # test_field (x = conflict area, should be chosen as 1)
    # ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 0, 2, 2, 2, 2, 0],
    # [0, 0, 1, 1, x, 2, 2, 2, 2, 0],
    # [0, 0, 1, 1, x, 2, 2, 2, 2, 0],
    # [0, 0, 0, 0, x, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    expected_field = np.zeros_like(test_field)
    expected_field[4:7, 3:5] = 1
    expected_field[4:7, 5:8] = 2
    # expected_field
    # ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 1, 1, 2, 2, 2, 0, 0],
    # [0, 0, 0, 1, 1, 2, 2, 2, 0, 0],
    # [0, 0, 0, 1, 1, 2, 2, 2, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    advected_field = advect_field_using_motion_vectors(
        test_field, test_y_flow, test_x_flow
    )
    err_msg = "Test failed: advected array is not equal to expected array."
    np.testing.assert_array_equal(advected_field, expected_field, err_msg)


# TODO: add tests for junk inputs
