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

    test_field3 = np.zeros((10, 10), dtype=int)

    feature1_mask_y = slice(8, 10)
    feature1_mask_x = slice(8, 10)
    test_field3[feature1_mask_y, feature1_mask_x] = 1

    # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

    test_field4 = np.zeros((10, 10), dtype=int)

    feature3_mask_y = slice(3, 7)
    feature3_mask_x = slice(5, 9)
    test_field4[feature3_mask_y, feature3_mask_x] = 3

    feature4_mask_y = slice(2, 4)
    feature4_mask_x = slice(3, 3)
    test_field4[feature4_mask_y, feature4_mask_x] = 4
    test_field4[3:5, 9] = 4

    # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 4],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 4],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 0],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    return test_field, test_field2, test_field3, test_field4


def test_generate_radial_mask():
    """
    Tests the geenrate_radial_mask function in FrameTracker
    """
    test_field, __, __, __ = construct_test_fields()

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
    test_field, __, __, __ = construct_test_fields()
    # Find centroid for value 1
    expected_centroid = np.array((4, 2.5))
    centroid = get_centroid(test_field, 1)
    np.testing.assert_array_equal(expected_centroid, centroid)


def test_overlap_histogram():
    """
    Tests whether the calculate_overlap_histogram produces the correct degree of overlap
    """

    test_field, test_field2, __, __ = construct_test_fields()

    # Expect to find overlap of 2/3 for label 1
    # For label 2, this is a bit tricker since each feature is a different shape
    # Overlap between fields is 9 pixels
    # overlap normalised by first field area = 9/12 = 3/4
    # overlap normalised by second field area = 9/16
    # full overlap = (3/4 + 9/16)/2 = 0.65625
    # Label 0 is reserved for the background, which should be set to 0
    ids = [0, 1, 2]
    expected_results = [0, 2 / 3, 0.65625]
    for id, expected_result in zip(ids, expected_results):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field, test_field2, feature_id=id
        )
        result = hist[id]
        err_msg = f"Test failed: overlap ({result}) not equal to expected overlap ({expected_result})."
        np.testing.assert_equal(result, expected_result, err_msg)


def test_overlap_histogram_with_nbhood():
    test_field, test_field2, __, __ = construct_test_fields()

    # Using a mask of radius 3 pixels around each feature, we now expect
    # to encompass all of the features with the same label in each field
    # Therefore, for label 1, expect an overlap of 1
    # We do not expect this for label 2, however, since this is a different
    # shape in each field. Now, there are 12 pixels that include the label 2
    # in the first input field, so the "overlap" is now considered to be 12.
    # Overlap normalised by first field area = 12/12 = 1
    # Overlap normalised by second field area = 12/16 = 3/4
    # full overlap = (1 + 3/4)/2 = 7/8 = 0.875
    # Label 0 is reserved for the background, which should be set to 0

    ids = [0, 1, 2]
    expected_results = [0, 1, 0.875]
    for id, expected_result in zip(ids, expected_results):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field, test_field2, feature_id=id, nbhood=3
        )
        result = hist[id]
        err_msg = f"Test failed: overlap ({result}) not equal to expected overlap ({expected_result})."
        np.testing.assert_equal(result, expected_result, err_msg)


def test_overlap_histogram_with_multiple_overlaps_and_different_labels():
    __, test_field2, __, test_field4 = construct_test_fields()

    # Use test_field4 as the advected field (containing multiple overlap labels)
    # and test_field2 as the current field that we want to find the overlap for
    # test_field4 was constructed so that it largely overlaps with test_field2
    # but that one label has a much more obvious similarity/overlap than the other
    hist = FrameTracker().calculate_overlap_histogram(
        test_field4, test_field2, feature_id=2, nbhood=0
    )

    # Expect overlap of 0.75 with label 3, and 0.125 with label 4
    # note this should not necessarily add to 1
    expected_hist = np.array([0, 0, 0, 0.75, 0.125])
    err_msg = f"Test failed: overlap ({hist}) not equal to expected overlap ({expected_hist})."
    np.testing.assert_array_equal(hist, expected_hist, err_msg)


def test_find_id_of_closest_overlap_with_single_label_overlap():
    # Test should find the same id is the best overlap between fields
    test_field1, test_field2, __, __ = construct_test_fields()

    for id in range(1, 3):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field1, test_field2, feature_id=id, nbhood=0
        )
        matching_id, others = FrameTracker().find_ids_of_closest_overlaps(
            hist, test_field1, test_field2, id
        )
        err_msg = "Test failed: matching id was not found to be equal to input id"
        np.testing.assert_equal(id, matching_id, err_msg)


def test_find_id_of_closest_overlap_with_no_overlap():
    # test should find no overlap between fields
    test_field1, __, test_field3, __ = construct_test_fields()

    hist = FrameTracker().calculate_overlap_histogram(
        test_field1, test_field3, feature_id=1, nbhood=0
    )
    matching_id, others = FrameTracker().find_ids_of_closest_overlaps(
        hist, test_field1, test_field3, 1
    )
    if matching_id is not None:
        raise ValueError(f"Test failed: Expected no matching id, got {matching_id}")

    if others is not None:
        raise ValueError(f"Test failed: Expected no other ids, got {others}")


def test_find_id_of_closest_overlap_with_multiple_overlaps_but_only_one_sufficient():
    # Using same data as test_overlap_histogram_with_multiple_overlaps_and_different_labels():
    # From this test, we expect the function to choose label 3 as the best overlap
    __, test_field2, __, test_field4 = construct_test_fields()
    hist = FrameTracker().calculate_overlap_histogram(
        test_field4, test_field2, feature_id=2, nbhood=0
    )
    val, others = FrameTracker().find_ids_of_closest_overlaps(
        hist, test_field4, test_field2, 2
    )
    expected_val = 3
    err_msg = f"Test failed: expected to find label {expected_val}, got {val}"
    np.testing.assert_equal(val, expected_val, err_msg)

    if others is not None:
        raise ValueError(f"Test failed: Expected no other ids, got {others}")


def test_find_id_of_closest_overlap_with_multiple_unequal_sufficient_overlaps():
    # can use the same test_fields as test_find_id_of_closest_overlap_with_multiple_overlaps_but_only_one_sufficient():
    # but can lower the overlap_threshold to make both feature sufficient
    # Since label 3 still has a much larger overlap it should still be chosen
    # However, need to set the overlap threshold to 0.1 for label 4 to be considered suitable
    # (see test_overlap_histogram_with_multiple_overlaps_and_different_labels() for overlap hist)
    __, test_field2, __, test_field4 = construct_test_fields()
    hist = FrameTracker().calculate_overlap_histogram(
        test_field4, test_field2, feature_id=2, nbhood=0
    )
    val, others = FrameTracker(overlap_threshold=0.1).find_ids_of_closest_overlaps(
        hist, test_field4, test_field2, 2
    )
    expected_val = 3
    err_msg = f"Test failed: expected to find label {expected_val}, got {val}"
    np.testing.assert_equal(val, expected_val, err_msg)

    expected_others = np.array(4)
    err_msg = (
        f"Test failed: expected to find other label {expected_others}, got {others}"
    )
    np.testing.assert_array_equal(others, expected_others, err_msg)


def test_find_id_of_closest_overlap_with_multiple_equally_sufficient_overlaps():
    # For multiple equally sufficient overlaps, need to construct a new test field
    # enforce overlap of 6 pixels with label 2 of test_field2 for each of the new labels
    # But, make feature 3 much larger than feature 4 so its centroid is further from the label 2 centroid

    __, test_field2, __, __ = construct_test_fields()
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

    test_field5 = np.zeros((10, 10), dtype=int)

    feature3_mask_y = slice(4, 9)
    feature3_mask_x = slice(1, 8)
    test_field5[feature3_mask_y, feature3_mask_x] = 3

    feature4_mask_y = slice(0, 6)
    feature4_mask_x = slice(8, 10)
    test_field5[feature4_mask_y, feature4_mask_x] = 4

    # test_field5
    # [[0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 3, 3, 3, 3, 3, 3, 3, 4, 4],
    # [0, 3, 3, 3, 3, 3, 3, 3, 4, 4],
    # [0, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    # [0, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    # [0, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    hist = FrameTracker().calculate_overlap_histogram(
        test_field5, test_field2, feature_id=2, nbhood=0
    )
    val, others = FrameTracker(overlap_threshold=0.3).find_ids_of_closest_overlaps(
        hist, test_field5, test_field2, 2
    )
    expected_val = 4
    err_msg = f"Test failed: expected to find label {expected_val}, got {val}"
    np.testing.assert_equal(val, expected_val, err_msg)

    expected_others = np.array(3)
    err_msg = (
        f"Test failed: expected to find other label {expected_others}, got {others}"
    )
    np.testing.assert_array_equal(others, expected_others, err_msg)


def test_find_id_of_closest_overlap_with_multiple_equally_sufficient_overlaps_and_equal_centroid_distances():
    # Now, construct symmetric features so that they have the same overlap and the same centroid distance
    # We expect code to therefore choose the feature with the lower value
    __, test_field2, __, __ = construct_test_fields()
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

    test_field6 = np.zeros((10, 10), dtype=int)

    feature3_mask_y = slice(4, 8)
    feature3_mask_x = slice(6, 8)
    test_field6[feature3_mask_y, feature3_mask_x] = 3

    feature4_mask_y = slice(2, 6)
    feature4_mask_x = slice(8, 10)
    test_field6[feature4_mask_y, feature4_mask_x] = 4
    # test_field6
    # ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 3, 3, 4, 4],
    # [0, 0, 0, 0, 0, 0, 3, 3, 4, 4],
    # [0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
    # [0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    hist = FrameTracker().calculate_overlap_histogram(
        test_field6, test_field2, feature_id=2, nbhood=0
    )
    val, others = FrameTracker(overlap_threshold=0.3).find_ids_of_closest_overlaps(
        hist, test_field6, test_field2, 2
    )

    expected_val = 3
    err_msg = f"Test failed: expected to find label {expected_val}, got {val}"
    np.testing.assert_equal(val, expected_val, err_msg)

    expected_others = np.array(4)
    err_msg = (
        f"Test failed: expected to find other label {expected_others}, got {others}"
    )
    np.testing.assert_array_equal(others, expected_others, err_msg)
