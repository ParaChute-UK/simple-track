import sys

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from src.FrameTracker import FrameTracker

import numpy as np


def test_overlap_histogram():
    """
    Tests whether the calculate_overlap_histogram produces the correct degree of overlap
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


if __name__ == "__main__":
    test_overlap_histogram()
