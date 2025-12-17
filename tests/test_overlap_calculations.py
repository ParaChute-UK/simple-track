import sys

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from src.FrameTracker import FrameTracker, generate_radial_mask, get_centroid

import numpy as np


def construct_test_fields():
    """
    Constructs test fields used for these tests.
    """
    test_field = np.zeros((10, 10), dtype=int)
    feature1_mask_y = slice(3, 6)
    feature1_mask_x = slice(2, 4)
    test_field[feature1_mask_y, feature1_mask_x] = 1

    feature2_mask_y = slice(3, 6)
    feature2_mask_x = slice(5, 9)
    test_field[feature2_mask_y, feature2_mask_x] = 2

    # test_field
    # ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 0, 2, 2, 2, 2, 0],
    # [0, 0, 1, 1, 0, 2, 2, 2, 2, 0],
    # [0, 0, 1, 1, 0, 2, 2, 2, 2, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    test_field2 = np.zeros((10, 10), dtype=int)

    # For feature 1, move this down by 1 pixel but keep the same shape
    feature1_mask_y = slice(4, 7)
    feature1_mask_x = slice(2, 4)
    test_field2[feature1_mask_y, feature1_mask_x] = 1

    # For feature 2, move this to the right by 1 pixel, but also expand its area down by
    # 1 row
    feature2_mask_y = slice(3, 7)
    feature2_mask_x = slice(6, 10)
    test_field2[feature2_mask_y, feature2_mask_x] = 2

    # test_field2
    # ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    # [0, 0, 1, 1, 0, 0, 2, 2, 2, 2],
    # [0, 0, 1, 1, 0, 0, 2, 2, 2, 2],
    # [0, 0, 1, 1, 0, 0, 2, 2, 2, 2],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    return test_field, test_field2


def test_overlap_histogram():
    """
    Tests whether the calculate_overlap_histogram produces the correct degree of overlap
    """

    test_field, test_field2 = construct_test_fields()

    # Expect to find overlap of 2/3 for label 1
    # For label 2, this is a bit tricker since each feature is a different shape
    # Overlap between fields is 9 pixels
    # overlap normalised by first field area = 9/12 = 3/4
    # overlap normalised by second field area = 9/16
    # full overlap = (3/4 + 9/16)/2 = 0.65625
    ids = [1, 2]
    expected_results = [2 / 3, 0.65625]
    for id, expected_result in zip(ids, expected_results):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field, test_field2, feature_id=id
        )
        result = hist[id]
        err_msg = f"Test failed: overlap ({result}) not equal to expected overlap ({expected_result})."
        np.testing.assert_equal(result, expected_result, err_msg)


def test_overlap_histogram_with_nbhood():
    test_field, test_field2 = construct_test_fields()

    # Using a mask of radius 3 pixels around each feature, we now expect
    # to encompass all of the features with the same label in each field
    # Therefore, for label 1, expect an overlap of 1
    # We do not expect this for label 2, however, since this is a different
    # shape in each field. Now, there are 12 pixels that include the label 2
    # in the first input field, so the "overlap" is now considered to be 12.
    # Overlap normalised by first field area = 12/12 = 1
    # Overlap normalised by second field area = 12/16 = 3/4
    # full overlap = (1 + 3/4)/2 = 7/8 = 0.875

    ids = [1, 2]
    expected_results = [1, 0.875]
    for id, expected_result in zip(ids, expected_results):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field, test_field2, feature_id=id, nbhood=3
        )
        result = hist[id]
        err_msg = f"Test failed: overlap ({result}) not equal to expected overlap ({expected_result})."
        np.testing.assert_equal(result, expected_result, err_msg)


def test_generate_radial_mask():
    """
    Tests the geenrate_radial_mask function in FrameTracker
    """
    test_field, __ = construct_test_fields()

    centroid = (6, 7)
    radius = 3

    expected_mask = np.zeros_like(test_field, dtype=bool)
    expected_mask[4:9, 5:] = True
    # ([[False, False, False, False, False, False, False, False, False, False],
    # [False, False, False, False, False, False, False, False, False, False],
    # [False, False, False, False, False, False, False, False, False, False],
    # [False, False, False, False, False, False, False, False, False, False],
    # [False, False, False, False, False,  True,  True,  True,  True, True],
    # [False, False, False, False, False,  True,  True,  True,  True, True],
    # [False, False, False, False, False,  True,  True,  True,  True, True],
    # [False, False, False, False, False,  True,  True,  True,  True, True],
    # [False, False, False, False, False,  True,  True,  True,  True, True],
    # [False, False, False, False, False, False, False, False, False False]])

    mask = generate_radial_mask(test_field, centroid, radius)
    err_msg = "Test failed: radial mask is not the same as expected."
    np.testing.assert_array_equal(mask, expected_mask, err_msg)


def test_get_centroid():
    test_field, __ = construct_test_fields()
    # Find centroid for value 1
    expected_centroid = np.array((4, 2.5))
    centroid = get_centroid(test_field, 1)
    np.testing.assert_array_equal(expected_centroid, centroid)


if __name__ == "__main__":
    test_generate_radial_mask()
    test_get_centroid()
    test_overlap_histogram()
    test_overlap_histogram_with_nbhood()
