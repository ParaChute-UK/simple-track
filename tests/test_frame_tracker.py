import datetime as dt

import numpy as np
import pytest

from simpletrack.exceptions import (
    ArrayError,
    ArrayShapeError,
    ArrayTypeError,
    FloatIDError,
    IDError,
    NegativeIDError,
    ZeroIDError,
)
from simpletrack.feature import Feature
from simpletrack.frame import Frame
from simpletrack.frame_tracker import (
    FrameTracker,
    advect_field_using_motion_vectors,
    generate_radial_mask,
    get_centroid,
)

zero_arr = np.zeros((10, 10), dtype=int)


@pytest.fixture
def construct_test_fields():
    """
    Constructs test fields used in this file.
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

    test_field5 = np.zeros((10, 10), dtype=int)

    feature3_mask_y = slice(3, 7)
    feature3_mask_x = slice(5, 9)
    test_field5[feature3_mask_y, feature3_mask_x] = 3

    test_field5[0:5, 9] = 4

    # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 4],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 4],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 0],
    #        [0, 0, 0, 0, 0, 3, 3, 3, 3, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    return test_field, test_field2, test_field3, test_field4, test_field5


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


@pytest.mark.parametrize(
    "field, yflow, xflow, expected_error",
    [
        ["not an array", zero_arr, zero_arr, ArrayTypeError],
        [zero_arr, "not an array", zero_arr, ArrayTypeError],
        [zero_arr, zero_arr, "not an array", ArrayTypeError],
        [np.zeros((5, 5), dtype=int), zero_arr, zero_arr, ArrayShapeError],
        [np.zeros((10, 10, 10), dtype=int), zero_arr, zero_arr, ArrayShapeError],
        [np.zeros(10, dtype=int), zero_arr, zero_arr, ArrayShapeError],
    ],
)
def test_advect_field_invalid_inputs(field, yflow, xflow, expected_error):
    try:
        __ = advect_field_using_motion_vectors(field, yflow, xflow)
    except expected_error:
        pass


def test_generate_radial_mask(construct_test_fields):
    """
    Tests the generate_radial_mask function in FrameTracker
    """
    test_field = construct_test_fields[0]
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


@pytest.mark.parametrize(
    "field, coord, mask_radius, expected_error",
    [
        [np.zeros(10, dtype=int), (5, 5), 3, ArrayShapeError],
        [np.zeros((10, 10, 10), dtype=int), (5, 5), 3, ArrayShapeError],
        [zero_arr, 5, 3, ArrayTypeError],
        [zero_arr, (5, 5, 5), 3, ArrayShapeError],
        [zero_arr, (5, 5), 3.4, TypeError],
        [zero_arr, (5, 5), -3, ValueError],
    ],
)
def test_generate_radial_mask_invalid_inputs(field, coord, mask_radius, expected_error):
    try:
        __ = generate_radial_mask(field, coord, mask_radius)
    except expected_error:
        pass


def test_get_centroid(construct_test_fields):
    """
    Test the get_centroid function using test field
    """
    test_field = construct_test_fields[0]
    # Find centroid for value 1
    expected_centroid = np.array((4, 2.5))
    centroid = get_centroid(test_field, 1)
    np.testing.assert_array_equal(expected_centroid, centroid)


@pytest.mark.parametrize(
    "field, value, expected_error",
    [
        [np.zeros(10, dtype=int), 3, ArrayShapeError],
        [np.zeros((10, 10, 10), dtype=int), 3, ArrayShapeError],
        [zero_arr, 3.4, FloatIDError],
        [zero_arr, -3, NegativeIDError],
    ],
)
def test_check_centroid_invalid_inputs(field, value, expected_error):
    try:
        __ = get_centroid(field, value)
    except expected_error:
        pass


def test_find_ids_of_closest_size_with_single_closest_size(construct_test_fields):
    """
    Test the find_id_of_closest_size function when there is a single candidate id
    with closest size to the target feature
    """
    test_field, test_field2 = construct_test_fields[0:2]

    closest_size_id = FrameTracker().find_ids_of_closest_size(
        test_field, test_field2, 1, [1, 2]
    )
    expected_id = [1]
    err_msg = f"Test failed: expected to find id {expected_id}, got {closest_size_id}"
    np.testing.assert_equal(closest_size_id, expected_id, err_msg)


def test_find_ids_of_closest_size_with_multiple_equally_closest_sizes(
    construct_test_fields,
):
    """
    Test the find_id_of_closest_size function when there are multiple candidate ids
    with equally close sizes to the target feature
    """
    test_field2 = construct_test_fields[1]

    test_field4 = np.zeros((10, 10), dtype=int)
    feature3_mask_y = slice(0, 4)
    feature3_mask_x = slice(0, 4)
    test_field4[feature3_mask_y, feature3_mask_x] = 3

    feature4_mask_y = slice(5, 9)
    feature4_mask_x = slice(5, 9)
    test_field4[feature4_mask_y, feature4_mask_x] = 4
    # array([[3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    #    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    #    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    #    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 4, 4, 4, 4, 0, 0],
    #    [0, 0, 0, 0, 4, 4, 4, 4, 0, 0],
    #    [0, 0, 0, 0, 4, 4, 4, 4, 0, 0],
    #    [0, 0, 0, 0, 4, 4, 4, 4, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    closest_size_ids = FrameTracker().find_ids_of_closest_size(
        test_field2, test_field4, 2, [3, 4]
    )
    expected_ids = [3, 4]
    err_msg = (
        f"Test failed: expected to find ids {expected_ids}, got {closest_size_ids}"
    )
    np.testing.assert_array_equal(closest_size_ids, expected_ids, err_msg)


def test_overlap_histogram(construct_test_fields):
    """
    Tests whether the calculate_overlap_histogram produces the correct degree of overlap
    """

    test_field, test_field2 = construct_test_fields[0:2]

    # Expect to find overlap of 2/3 for label 1
    # And overlap = 12/16 = 0.75 for label 2
    # Label 0 is reserved for the background, which should be set to 0
    # But this is not tested here since this would throw an IDError
    ids = [1, 2]
    expected_results = [2 / 3, 0.75]
    for id, expected_result in zip(ids, expected_results):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field, test_field2, feature_id=id
        )
        result = hist[id]
        err_msg = f"Test failed: overlap ({result}) not equal to expected overlap ({expected_result})."
        np.testing.assert_equal(result, expected_result, err_msg)


def test_overlap_histogram_with_nbhood(construct_test_fields):
    """
    Tests whether the calculate_overlap_histogram produces the correct degree of overlap
    when a nbhood is used to expand the mask around the first input.
    """

    test_field, test_field2 = construct_test_fields[0:2]

    # Using a mask of radius 3 pixels around each feature, we now expect
    # to encompass all of the features with the same label in each field
    # Therefore, for label 1 and 2, expect an overlap of 1 (full overlap)
    ids = [1, 2]
    expected_results = [1, 1]
    for id, expected_result in zip(ids, expected_results):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field, test_field2, feature_id=id, nbhood=3
        )
        result = hist[id]
        err_msg = f"Test failed: overlap ({result}) not equal to expected overlap ({expected_result})."
        np.testing.assert_equal(result, expected_result, err_msg)


def test_overlap_histogram_with_multiple_overlaps_and_different_labels(
    construct_test_fields,
):
    """
    Test calculate_overlap_histogram when there are multiple overlapping labels
    for the requested feature id
    """
    test_field2 = construct_test_fields[1]
    test_field4 = construct_test_fields[3]

    # Use test_field4 as the advected field (containing multiple overlap labels)
    # and test_field2 as the current field that we want to find the overlap for
    # test_field4 was constructed so that it largely overlaps with test_field2
    # but that one label has a much more obvious similarity/overlap than the other
    hist = FrameTracker().calculate_overlap_histogram(
        test_field4, test_field2, feature_id=2, nbhood=0
    )

    # Expect overlap of 0.75 with label 3, and 1 with label 4
    expected_hist = np.array([0, 0, 0, 0.75, 1])
    err_msg = f"Test failed: overlap ({hist}) not equal to expected overlap ({expected_hist})."
    np.testing.assert_array_equal(hist, expected_hist, err_msg)


@pytest.mark.parametrize(
    "advected_field, current_field, feature_id, nbhood, expected_error",
    [
        [np.zeros((10), dtype=int), zero_arr, 1, 0, ArrayShapeError],
        [
            np.zeros((10, 10, 10), dtype=int),
            np.zeros((10, 10, 10)),
            1,
            0,
            ArrayShapeError,
        ],
        [np.zeros((10), dtype=int), np.zeros((10), dtype=int), 1, 0, ArrayShapeError],
        [np.zeros((5, 10), dtype=int), zero_arr, 1, 0, ArrayShapeError],
        [np.zeros((10, 10), dtype=float), np.zeros((10, 10)), 1, 0, ArrayTypeError],
        [zero_arr, zero_arr, -1, 0, NegativeIDError],
        [zero_arr, zero_arr, 1.5, 0, FloatIDError],
        [zero_arr, zero_arr, 0, 0, ZeroIDError],
        [zero_arr, zero_arr, 1, 0.5, TypeError],
        [zero_arr, zero_arr, 1, -1, ValueError],
    ],
)
def test_overlap_histogram_invalid_inputs(
    advected_field, current_field, feature_id, nbhood, expected_error
):
    try:
        FrameTracker().calculate_overlap_histogram(
            advected_field, current_field, feature_id, nbhood
        )
    except expected_error:
        pass


def test_find_id_of_closest_overlap_with_single_label_overlap(construct_test_fields):
    """
    Test the find_ids_of_closest_overlaps function when there is a single overlapping label
    """
    # Test should find the same id is the best overlap between fields
    test_field1, test_field2 = construct_test_fields[0:2]

    for id in range(1, 3):
        hist = FrameTracker().calculate_overlap_histogram(
            test_field1, test_field2, feature_id=id, nbhood=0
        )
        matching_id, others = FrameTracker().find_ids_of_closest_overlaps(
            hist, test_field1, test_field2, id
        )
        err_msg = "Test failed: matching id was not found to be equal to input id"
        np.testing.assert_equal(id, matching_id, err_msg)


def test_find_id_of_closest_overlap_with_no_overlap(construct_test_fields):
    """
    Test the find_ids_of_closest_overlaps function when there is no overlapping label
    """
    # test should find no overlap between fields
    test_field1, __, test_field3, __, __ = construct_test_fields
    test_field3 = construct_test_fields[2]

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


def test_find_id_of_closest_overlap_with_multiple_overlaps_but_only_one_sufficient(
    construct_test_fields,
):
    """
    Test the find_ids_of_closest_overlaps function when there are multiple overlapping labels
    but only one meets the overlap threshold
    """
    # Using same data as test_overlap_histogram_with_multiple_overlaps_and_different_labels():
    # From this test, we expect the function to choose label 3 as the best overlap
    __, test_field2, __, __, test_field5 = construct_test_fields
    hist = FrameTracker().calculate_overlap_histogram(
        test_field5, test_field2, feature_id=2, nbhood=0
    )
    val, others = FrameTracker().find_ids_of_closest_overlaps(
        hist, test_field5, test_field2, 2
    )
    expected_val = 3
    err_msg = f"Test failed: expected to find label {expected_val}, got {val}"
    np.testing.assert_equal(val, expected_val, err_msg)

    if others is not None:
        raise ValueError(f"Test failed: Expected no other ids, got {others}")


def test_find_id_of_closest_overlap_with_multiple_unequal_sufficient_overlaps(
    construct_test_fields,
):
    """
    Test the find_ids_of_closest_overlaps function when there are multiple overlapping labels
    that each exceed the overlap threshold, but one is clearly more overlapping than the other
    """
    # can use the same test_fields as test_find_id_of_closest_overlap_with_multiple_overlaps_but_only_one_sufficient():
    # but can lower the overlap_threshold to make both feature sufficient
    # Since label 3 still has a much larger overlap it should still be chosen
    # However, need to set the overlap threshold to 0.1 for label 4 to be considered suitable
    # (see test_overlap_histogram_with_multiple_overlaps_and_different_labels() for overlap hist)
    __, test_field2, __, test_field4, __ = construct_test_fields
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


def test_find_id_of_closest_overlap_with_multiple_equally_sufficient_overlaps(
    construct_test_fields,
):
    """
    Test the find_ids_of_closest_overlaps function when there are multiple overlapping labels
    that each exceed the overlap threshold by the same degree, but one is closer in centroid distance
    """
    # For multiple equally sufficient overlaps, need to construct a new test field
    # enforce overlap of 6 pixels with label 2 of test_field2 for each of the new labels
    # But, make feature 3 much larger than feature 4 so its centroid is further from the label 2 centroid

    test_field2 = construct_test_fields[1]
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
    feature3_mask_x = slice(4, 8)
    test_field5[feature3_mask_y, feature3_mask_x] = 3

    feature4_mask_y = slice(0, 6)
    feature4_mask_x = slice(8, 10)
    test_field5[feature4_mask_y, feature4_mask_x] = 4

    # test_field5
    # [[0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    # [0, 0, 0, 0, 3, 3, 3, 3, 4, 4],
    # [0, 0, 0, 0, 3, 3, 3, 3, 4, 4],
    # [0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
    # [0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
    # [0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    hist = FrameTracker().calculate_overlap_histogram(
        test_field5, test_field2, feature_id=2, nbhood=0
    )
    print(hist)
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


def test_find_id_of_closest_overlap_with_multiple_equally_sufficient_overlaps_and_equal_centroid_distances(
    construct_test_fields,
):
    """
    Test the find_ids_of_closest_overlaps function when there are multiple overlapping labels
    that each exceed the overlap threshold by the same degree, AND have the same centroid distance.
    This should therefore return the label with the lowest value.
    """
    # Now, construct symmetric features so that they have the same overlap and the same centroid distance
    # We expect code to therefore choose the feature with the lower value
    test_field2 = construct_test_fields[1]
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


@pytest.mark.parametrize(
    "overlap_hist, advected_field, current_field, feature_id, expected_error",
    [
        [zero_arr, zero_arr, zero_arr, 1, ArrayShapeError],
        [np.arange(-5, 5), zero_arr, zero_arr, 1, ArrayTypeError],
        [
            np.arange(5),
            np.zeros((10), dtype=int),
            np.zeros((10), dtype=int),
            1,
            ArrayError,
        ],
        [
            np.arange(5),
            np.zeros((10, 10, 10), dtype=int),
            np.zeros((10, 10, 10), dtype=int),
            1,
            ArrayShapeError,
        ],
        [np.arange(5), np.zeros((5, 5), dtype=int), zero_arr, 1, ArrayShapeError],
        [np.arange(5), zero_arr, zero_arr, -1, NegativeIDError],
        [np.arange(5), zero_arr, zero_arr, 1.5, FloatIDError],
        [np.arange(5), zero_arr, zero_arr, 0, ZeroIDError],
    ],
)
def test_find_ids_closest_overlap_invalid_inputs(
    overlap_hist, advected_field, current_field, feature_id, expected_error
):
    try:
        FrameTracker().find_ids_of_closest_overlaps(
            overlap_hist, advected_field, current_field, feature_id
        )
    except expected_error:
        pass


def test_identify_parent_and_child_features_with_complete_overlap():
    """
    Test the identify_parent_and_child_features method of FrameTracker
    when the parent feature completely overlaps with one of the current features.
    """

    # Test the identify parent and child features function by making a big object in the advected field
    # that overlaps with multiple other smaller objects in the current field
    # It should choose the big object of a similar-ish size as the parent feature.
    test_arr1 = np.zeros((10, 10), dtype=int)
    test_arr2 = np.zeros((10, 10), dtype=int)

    y_slice = slice(1, 9)
    x_slice = slice(1, 9)

    test_arr1[y_slice, x_slice] = 1

    # Populate test_arr2 with multiple smaller features that overlap with feature 1 of test_arr1
    # This should intuitively choose feature 3 as the parent feature since it is the largest feature
    # and it is fully overlapping.
    # Feature 1 is smaller and only partially contained within feature 1 of test_arr1
    # Feature 2 is also smaller but is fully contained within feature 1 of test_arr1

    feature1_y_slice = slice(0, 4)
    feature1_x_slice = slice(7, 9)

    feature2_y_slice = slice(6, 9)
    feature2_x_slice = feature1_x_slice

    feature3_y_slice = slice(1, 9)
    feature3_x_slice = slice(1, 6)

    test_arr2[feature1_y_slice, feature1_x_slice] = 1
    test_arr2[feature2_y_slice, feature2_x_slice] = 2
    test_arr2[feature3_y_slice, feature3_x_slice] = 3

    # For this test, we assume that features 1,2,3 in test_arr2 were matched with feature 1 in test_arr1 and
    # were therefore given a provisional id of 1. Therefore, we want the code to correctly identify feautre 3
    # as the parent, and features 1 and 2 as the children

    # First, setup the current frame matching feature list
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    feature_ids = [1, 2, 3]
    feature_list = [
        Feature(
            id=fid, feature_coords=np.array(np.where(test_arr2 == fid)), time=feature_dt
        )
        for fid in feature_ids
    ]

    # Call the method
    parent_feature, child_features = FrameTracker().identify_parent_and_child_features(
        parent_id=1,
        matching_features=feature_list,
        advected_feature_field=test_arr1,
        current_feature_field=test_arr2,
    )

    assert parent_feature.id == 3
    assert len(child_features) == 2
    assert all(cf.id in [1, 2] for cf in child_features)


def test_identify_parent_and_child_features_with_partial_overlap():
    """
    Test the identify_parent_and_child_features method of FrameTracker
    when the parent feature partially overlaps with one of the current features.
    """

    test_arr1 = np.zeros((10, 10), dtype=int)
    test_arr2 = np.zeros((10, 10), dtype=int)

    y_slice = slice(1, 9)
    x_slice = slice(1, 9)

    test_arr1[y_slice, x_slice] = 1

    # Populate test_arr2 with multiple smaller features that overlap with feature 1 of test_arr1
    # This should intuitively choose feature 3 as the parent feature since it is the largest feature
    # despite the fact it is only partially overlapping.
    # Feature 1 is smaller and only partially contained within feature 1 of test_arr1
    # Feature 2 is also smaller but is fully contained within feature 1 of test_arr1

    feature1_y_slice = slice(0, 4)
    feature1_x_slice = slice(7, 9)

    feature2_y_slice = slice(6, 9)
    feature2_x_slice = feature1_x_slice

    feature3_y_slice = slice(1, 9)
    feature3_x_slice = slice(0, 5)

    test_arr2[feature1_y_slice, feature1_x_slice] = 1
    test_arr2[feature2_y_slice, feature2_x_slice] = 2
    test_arr2[feature3_y_slice, feature3_x_slice] = 3

    # First, setup the current frame matching feature list
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    feature_ids = [1, 2, 3]
    feature_list = [
        Feature(
            id=fid, feature_coords=np.array(np.where(test_arr2 == fid)), time=feature_dt
        )
        for fid in feature_ids
    ]

    # Call the method
    parent_feature, child_features = FrameTracker().identify_parent_and_child_features(
        parent_id=1,
        matching_features=feature_list,
        advected_feature_field=test_arr1,
        current_feature_field=test_arr2,
    )

    assert parent_feature.id == 3
    assert len(child_features) == 2
    assert all(cf.id in [1, 2] for cf in child_features)


def test_identify_parent_and_child_features_with_no_overlap():
    """
    Test the identify_parent_and_child_features method of FrameTracker
    when the parent feature has no overlap with any of the current features.
    This should raise a ValueError.
    """

    # Create advected and current feature fields with no overlapping features
    test_arr1 = np.zeros((10, 10), dtype=int)
    test_arr2 = np.zeros((10, 10), dtype=int)

    # Feature in advected field
    test_arr1[1:5, 1:5] = 1

    # Features in current field that do not overlap with feature 1 in advected field
    test_arr2[6:9, 6:9] = 2
    test_arr2[0:2, 7:9] = 3

    # Setup the current frame matching feature list
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    feature_ids = [2, 3]
    feature_list = [
        Feature(
            id=fid, feature_coords=np.array(np.where(test_arr2 == fid)), time=feature_dt
        )
        for fid in feature_ids
    ]

    # Call the method and expect a ValueError
    try:
        parent_feature, child_features = (
            FrameTracker().identify_parent_and_child_features(
                parent_id=1,
                matching_features=feature_list,
                advected_feature_field=test_arr1,
                current_feature_field=test_arr2,
            )
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")


def test_identify_parent_and_child_features_with_missing_parent_id():
    """
    Test the identify_parent_and_child_features method of FrameTracker
    when the parent_id is not present in the advected feature field.
    This should raise a ValueError.
    """

    # Create advected and current feature fields
    test_arr1 = np.zeros((10, 10), dtype=int)
    test_arr2 = np.zeros((10, 10), dtype=int)

    # Feature in advected field
    test_arr1[1:5, 1:5] = 1

    # Features in current field
    test_arr2[1:4, 1:4] = 2
    test_arr2[6:9, 6:9] = 3

    # Setup the current frame matching feature list
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    feature_ids = [2, 3]
    feature_list = [
        Feature(
            id=fid, feature_coords=np.array(np.where(test_arr2 == fid)), time=feature_dt
        )
        for fid in feature_ids
    ]

    # Call the method with a parent_id that does not exist in advected field and expect a ValueError
    try:
        parent_feature, child_features = (
            FrameTracker().identify_parent_and_child_features(
                parent_id=2,
                matching_features=feature_list,
                advected_feature_field=test_arr1,
                current_feature_field=test_arr2,
            )
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")


def test_check_accreted_feature_ids_are_not_provisional_ids():
    frame_tracker = FrameTracker()
    test_frame = Frame()

    test_time = dt.datetime.now()
    test_coords = np.array(((1, 1),))
    test_features = {id: Feature(id, test_coords, test_time) for id in range(1, 4)}

    # Add an accreted feature id to Feature 2 that is already present in the Frame
    test_features[2].accreted = 3

    # Add an accreted feature id to Feature 3 that is not present in the Frame
    test_features[3].accreted = 10

    # Add provisional ids to each feature
    for feature in test_features.values():
        feature.provisional_id = feature.id

    # Add features to the Frame
    test_frame.features = test_features

    # Now, run the method which should remove this id from the accreted list
    frame_tracker.check_accreted_feature_ids_are_not_provisional_ids(test_frame)

    # Check that this correctly removed the accreted id for Feature 2
    assert test_frame.get_feature(2).accreted is None

    # Check that this correctly kept the accreted id for Feature 2
    assert test_frame.get_feature(3).accreted == [10]


def test_check_accreted_feature_ids_are_not_provisional_ids_valid():
    frame_tracker = FrameTracker()
    test_frame = Frame()

    test_time = dt.datetime.now()
    test_coords = np.array(((1, 1),))
    test_features = {id: Feature(id, test_coords, test_time) for id in range(1, 4)}

    # Add accreted feature ids to each feature that are not present in the Frame
    test_features[1].accreted = 10
    test_features[2].accreted = 11
    test_features[3].accreted = 1

    # Add provisional ids to each feature
    for feature in test_features.values():
        feature.provisional_id = feature.id

    # Add features to the Frame
    test_frame.features = test_features

    # Now, run the method which should not remove any accreted ids since they are all valid
    frame_tracker.check_accreted_feature_ids_are_not_provisional_ids(test_frame)

    # Check that all accreted ids are still present where appropriate
    assert test_frame.get_feature(1).accreted == [10]
    assert test_frame.get_feature(2).accreted == [11]
    # If empty list is passed to feature.accreted setter, this is replaced with None
    assert test_frame.get_feature(3).accreted is None


def test_check_accreted_feature_ids_are_not_provisional_ids_with_no_provisional_ids():
    frame_tracker = FrameTracker()
    test_frame = Frame()

    test_time = dt.datetime.now()
    test_coords = np.array(((1, 1),))
    test_features = {id: Feature(id, test_coords, test_time) for id in range(1, 4)}

    # Add accreted feature ids to each feature that are not present in the Frame
    test_features[1].accreted = 10
    test_features[2].accreted = 11
    test_features[3].accreted = 1

    # Add features to the Frame
    test_frame.features = test_features

    # Now, run the method which should not remove any accreted ids since they are all valid
    frame_tracker.check_accreted_feature_ids_are_not_provisional_ids(test_frame)

    # Check that all accreted ids are still present where appropriate
    assert test_frame.get_feature(1).accreted == [10]
    assert test_frame.get_feature(2).accreted == [11]
    # If empty list is passed to feature.accreted setter, this is replaced with None
    assert test_frame.get_feature(3).accreted == [1]


def test_identify_unmatched_features_in_prev_frame_valid_inputs():
    frame_tracker = FrameTracker()
    test_time = dt.datetime.now()

    current_frame = Frame()
    prev_frame = Frame()

    # Create features for the previous frame
    prev_features = {
        id: Feature(id, np.array(((1, 1),)), test_time) for id in range(1, 6)
    }
    prev_frame.features = prev_features

    # Create features for the current frame, where some features are unmatched with the previous frame
    current_features = {
        id: Feature(id, np.array(((1, 1),)), test_time) for id in range(3, 8)
    }
    current_frame.features = current_features

    # Now, run the method to find unmatched features
    frame_tracker.identify_unmatched_features_in_prev_frame(prev_frame, current_frame)

    # Check the unmatched features in previous frame are correctly identified
    expected_unmatched_ids = [1, 2]
    actual_unmatched_ids = [
        feature.id
        for feature in prev_frame.features.values()
        if feature.is_final_timstep()
    ]

    assert set(expected_unmatched_ids) == set(actual_unmatched_ids)


def test_identify_unmatched_features_in_prev_frame_invalid_inputs():
    frame_tracker = FrameTracker()
    try:
        frame_tracker.identify_unmatched_features_in_prev_frame(
            "Not a frame", "Not a frame"
        )
    except TypeError:
        pass


def test__get_overlap_sizes_same_feature_size():
    region1 = np.zeros((10, 10))
    region2 = np.zeros((10, 10))
    region1[2:8, 2:8] = 1  # 6x6 region
    region2[2:8, 2:8] = 2  # 6x6 region, fully overlapping
    overlap = FrameTracker()._get_overlap_sizes(region1, region2, 1, [2])
    assert overlap == [36]


def test__get_overlap_sizes_full_overlap_different_feature_size():
    region1 = np.zeros((10, 10))
    region2 = np.zeros((10, 10))
    region1[2:8, 2:8] = 1
    region2[3:7, 3:7] = 2
    region2[8:9, 8:9] = 3
    overlap = FrameTracker()._get_overlap_sizes(region1, region2, 1, [2, 3])
    assert overlap == [16, 0]


def test__get_overlap_sizes_partial_overlap():
    region1 = np.zeros((10, 10))
    region2 = np.zeros((10, 10))
    region1[2:8, 2:8] = 1
    region2[6:9, 6:9] = 2  # Only a 2x2 overlap with region1
    region2[8:9, 8:9] = 3
    overlap = FrameTracker()._get_overlap_sizes(region1, region2, 1, [2, 3])
    assert overlap == [4, 0]


def test__get_overlap_sizes_no_overlap():
    region1 = np.zeros((10, 10))
    region2 = np.zeros((10, 10))
    region1[2:5, 2:5] = 1
    region2[5:7, 5:7] = 2
    region2[8:9, 8:9] = 3
    overlap = FrameTracker()._get_overlap_sizes(region1, region2, 1, [2, 3])
    assert overlap == [0, 0]


# TODO: come back to this after considering best way of implementing nbhood in the code
# def test__get_overlap_sizes_no_overlap_using_nbhood():
#     region1 = np.zeros((10, 10))
#     region2 = np.zeros((10, 10))
#     region1[2:5, 2:5] = 1
#     region2[5:7, 5:7] = 2
#     region2[8:9, 8:9] = 3
#     overlap = FrameTracker()._get_overlap_sizes(region1, region2, 1, [2, 3], nbhood=2)
#     assert overlap == [0, 0]


def test_identify_parent_and_child_features_valid():
    current_field = np.zeros((10, 10), dtype=int)
    advected_field = np.zeros((10, 10), dtype=int)

    # Create a single parent feature in the advected field
    advected_field[3:7, 3:7] = 1

    # Create multiple features in current field, with one being most
    # obvious overlap
    current_field[6:9, 6:9] = 10
    current_field[3:7, 0:5] = 20  # This should be chosen as parent
    current_field[8:9, 6:9] = 30

    # Create list of matching_features for advected fields
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    matching_features = [
        Feature(
            id=id,
            feature_coords=np.array(np.where(current_field == id)),
            time=feature_dt,
        )
        for id in range(10, 40, 10)
    ]

    parent_feature, child_features = FrameTracker().identify_parent_and_child_features(
        parent_id=1,
        matching_features=matching_features,
        advected_feature_field=advected_field,
        current_feature_field=current_field,
    )
    child_ids = [cf.id for cf in child_features]
    assert parent_feature.id == 20
    assert set(child_ids) == set((10, 30))


# TODO: when nbhood has been properly thought through.
# def test_identify_parent_and_child_features_where_halo_is_required():
#     pass


def test_identify_parent_and_child_features_with_no_matching_features():
    current_field = np.zeros((10, 10), dtype=int)
    advected_field = np.zeros((10, 10), dtype=int)

    # Create a single parent feature in the advected field
    advected_field[0:2, 0:2] = 1  # Place this feature in the corner to avoid overlap

    # Create multiple features in current field
    current_field[6:9, 6:9] = 10
    current_field[3:7, 0:5] = 20
    current_field[8:9, 6:9] = 30

    # Create list of matching_features for advected fields
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    matching_features = [
        Feature(
            id=id,
            feature_coords=np.array(np.where(current_field == id)),
            time=feature_dt,
        )
        for id in range(10, 40, 10)
    ]
    try:
        parent_feature, child_features = (
            FrameTracker().identify_parent_and_child_features(
                parent_id=1,
                matching_features=matching_features,
                advected_feature_field=advected_field,
                current_feature_field=current_field,
            )
        )
    except ValueError:
        pass


def test_identify_parent_and_child_features_with_same_overlap_and_different_centroid_distances():
    current_field = np.zeros((10, 10), dtype=int)
    advected_field = np.zeros((10, 10), dtype=int)

    # Create a single parent feature in the advected field
    advected_field[2:9, 2:9] = 1

    # Create multiple features in current field with same overlap
    current_field[4:6, 3:6] = 10  # This should be chosen as parent due to centroid
    current_field[7:9, 6:9] = 20

    # Create list of matching_features for advected fields
    feature_dt = dt.datetime.now()  # this is unimportant for this test
    matching_features = [
        Feature(
            id=id,
            feature_coords=np.array(np.where(current_field == id)),
            time=feature_dt,
        )
        for id in range(10, 30, 10)
    ]

    parent_feature, child_features = FrameTracker().identify_parent_and_child_features(
        parent_id=1,
        matching_features=matching_features,
        advected_feature_field=advected_field,
        current_feature_field=current_field,
    )
    child_ids = [cf.id for cf in child_features]
    assert parent_feature.id == 10
    assert child_ids == [20]


current_field = np.zeros((10, 10), dtype=int)
advected_field = np.zeros((10, 10), dtype=int)
# Create list of matching_features for advected fields
feature_dt = dt.datetime.now()  # this is unimportant for this test
matching_features = [
    Feature(
        id=id,
        feature_coords=np.array(np.where(current_field == id)),
        time=feature_dt,
    )
    for id in range(10, 30, 10)
]


@pytest.mark.parametrize(
    "parent_id, matching_features, advected_field, current_field, expected_error",
    [
        [1, matching_features, "not an array", current_field, ArrayTypeError],
        [1, matching_features, advected_field, "not an array", ArrayTypeError],
        # ndim != 2
        [
            1,
            matching_features,
            advected_field,
            np.array((1), dtype=int),
            ArrayShapeError,
        ],
        # unequal shape
        [
            1,
            matching_features,
            advected_field,
            np.array((1, 1), dtype=int),
            ArrayShapeError,
        ],
        [1, ["not a feature list"], advected_field, current_field, TypeError],
        ["not an int", matching_features, advected_field, current_field, IDError],
    ],
)
def test_identify_parent_and_child_features_invalid_inputs(
    parent_id, matching_features, advected_field, current_field, expected_error
):

    # Check non-array inputs
    try:
        parent_feature, child_features = (
            FrameTracker().identify_parent_and_child_features(
                parent_id=parent_id,
                matching_features=matching_features,
                advected_feature_field=advected_field,
                current_feature_field=current_field,
            )
        )
    except expected_error:
        pass


current_field = np.zeros((10, 10), dtype=int)
advected_field = np.zeros((10, 10), dtype=int)
# Create a single parent feature in the advected field
current_field[2:8, 2:8] = 1
current_field[8:9, 8:9] = 2

# Create multiple features in current field with same overlap
advected_field[1:6, 3:6] = 10
advected_field[7:9, 6:9] = 20

closest_id_tests = [
    [current_field, advected_field, 1, [10, 20], [10]],
    [current_field, advected_field, 2, [10, 20], [20]],
    [current_field, "not an array", 1, [10, 20], ArrayTypeError],
    ["not an array", advected_field, 1, [10, 20], ArrayTypeError],
    [current_field, advected_field, 1.4, [10, 20], FloatIDError],
    [current_field, advected_field, -1, [10, 20], NegativeIDError],
    [current_field, advected_field, 1, [-10, 20], NegativeIDError],
    [current_field, advected_field, 1, [10.5, 20], FloatIDError],
]


@pytest.mark.parametrize(
    "field_with_id, field_to_search, target_feature_id, candidate_ids, expected_result",
    closest_id_tests,
)
def test_find_id_of_closest_size(
    field_with_id, field_to_search, target_feature_id, candidate_ids, expected_result
):
    try:
        result = FrameTracker().find_ids_of_closest_size(
            field_with_id, field_to_search, target_feature_id, candidate_ids
        )
        assert result == expected_result
    except expected_result:
        pass


@pytest.mark.parametrize(
    "field_with_id, field_to_search, target_feature_id, candidate_ids, expected_result",
    closest_id_tests,
)
def test_find_id_of_closest_centroid(
    field_with_id, field_to_search, target_feature_id, candidate_ids, expected_result
):
    try:
        result = FrameTracker().find_ids_of_closest_centroid(
            field_with_id, field_to_search, target_feature_id, candidate_ids
        )
        assert result == expected_result
    except expected_result:
        pass
