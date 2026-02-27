import sys
import pytest
import numpy as np
import datetime as dt

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Documents/Code/simple-track/src"
)
from frame import Frame, label_features
from feature import Feature


def test_label_features_valid_inputs():
    test_field = np.zeros((10, 10))
    test_field[1:3, 1:3] = 5
    expected_field = test_field.copy()
    expected_field[1:3, 1:3] = 1
    feature_field = label_features(test_field, min_area=3, threshold=3)
    np.testing.assert_array_equal(feature_field, expected_field)


def test_label_features_with_values_under_threshold():
    test_field = np.zeros((10, 10))
    test_field[1:3, 1:3] = 5
    expected_field = test_field.copy()
    test_field[6:9, 6:9] = 2
    expected_field[1:3, 1:3] = 1
    feature_field = label_features(test_field, min_area=3, threshold=3)
    np.testing.assert_array_equal(feature_field, expected_field)


def test_label_features_with_values_under_min_area():
    test_field = np.zeros((10, 10))
    test_field[1:4, 1:4] = 5
    expected_field = test_field.copy()
    test_field[6:7, 6:7] = 5
    expected_field[1:4, 1:4] = 1
    feature_field = label_features(test_field, min_area=5, threshold=3)
    np.testing.assert_array_equal(feature_field, expected_field)


test_field = np.zeros((10, 10))


@pytest.mark.parametrize(
    "field, min_area, threshold, under_threshold, expected_result",
    [
        [test_field, "not a min area", 2, True, ValueError],
        [test_field, 2, "not a threshold", True, ValueError],
        [test_field, 2, 2, "not a bool", TypeError],
        ["not a field", 2, 2, True, TypeError],
        [np.zeros((10, 10, 10)), 2, 2, True, ValueError],
    ],
)
def test_label_features_invalid_inputs(
    field, min_area, threshold, under_threshold, expected_result
):
    try:
        label_features(field, min_area, threshold, under_threshold)
    except expected_result:
        pass


def test_populate_features_valid_feature_field():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_feature_field = test_field.copy()
    test_feature_field[3:5, 3:5] = 1
    test_feature_field[6:9, 6:9] = 2

    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    expected_dict = {
        1: Feature(1, np.where(test_feature_field == 1), test_time),
        2: Feature(2, np.where(test_feature_field == 2), test_time),
    }

    assert test_frame.features == expected_dict


def test_populate_features_invalid_negative_features():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_feature_field = test_field.copy()
    test_feature_field[3:5, 3:5] = -1
    test_feature_field[6:9, 6:9] = 2

    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    try:
        test_frame.populate_features()
    except ValueError:
        pass


def test_populate_features_invalid_float_features():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_feature_field = test_field.copy()
    test_feature_field[3:5, 3:5] = 0.1
    test_feature_field[6:9, 6:9] = 2

    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    try:
        test_frame.populate_features()
    except TypeError:
        pass


def test_populate_features_with_no_feature_field():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_frame.populate_features()

    if len(test_frame.get_features()) != 0:
        raise TypeError(
            f"No features expected for this test, got {len(test_frame.get_features())}"
        )


def test_assign_displacements_valid_inputs():
    test_frame = Frame()
    y_flow = np.ones((10, 10))
    x_flow = np.ones((10, 10))
    # Make zero flow on one half of domain
    y_flow[:, :5] = 0
    x_flow[:, :5] = 0

    test_features = np.zeros((10, 10))
    test_features[2:4, 2:4] = 1
    test_features[6:9, 6:9] = 2

    test_frame.feature_field = test_features
    test_frame.populate_features()
    test_frame.assign_displacements(y_flow, x_flow)

    expected_displacements = {
        1: (0.0, 0.0),  # No flow in this region
        2: (1.0, 1.0),  # Flow of 1 in both directions
    }

    for feature_id, expected_disp in expected_displacements.items():
        feature = test_frame.get_feature(feature_id)
        assert feature.dydx == expected_disp


def test_assign_displacements_no_features_loaded():
    test_frame = Frame()
    y_flow = np.ones((10, 10))
    x_flow = np.ones((10, 10))

    try:
        test_frame.assign_displacements(y_flow, x_flow)
    except Exception:
        pass


def test_assign_displacements_invalid_inputs():
    test_frame = Frame()
    test_features = np.zeros((10, 10))
    test_features[2:4, 2:4] = 1
    test_features[6:9, 6:9] = 2

    test_frame.feature_field = test_features
    test_frame.populate_features()
    y_flow = "not a flow array"
    x_flow = np.ones((10, 10))

    try:
        test_frame.assign_displacements(y_flow, x_flow)
    except TypeError:
        pass
