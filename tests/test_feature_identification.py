import sys
from pathlib import Path
import pytest
import numpy as np
import datetime as dt

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from simple_track import SimpleTrack
from load import BaseLoader
from frame import Frame, label_features


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
