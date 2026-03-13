import datetime as dt
import sys

import numpy as np
import pytest

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Documents/Code/simple-track/src"
)
from feature import Feature
from frame import FeaturesNotFoundError, Frame, Timeline, label_features
from utils import ArrayShapeError, ArrayTypeError, FloatIDError, NegativeIDError


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
        ["not a field", 2, 2, True, ArrayTypeError],
        [np.zeros((10, 10, 10)), 2, 2, True, ArrayShapeError],
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


def test_populate_features_sets_extreme_property():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_feature_field = test_field.copy()
    test_feature_field[3:5, 3:5] = 1
    test_feature_field[6:9, 6:9] = 2

    test_raw_field = test_field.copy()
    test_raw_field[3:5, 3:5] = 10
    test_raw_field[6:9, 6:9] = 20
    # Set another higher maximum within the field to check extreme
    # only picks up the values within the feature mask
    test_raw_field[0:2, 0:2] = 100

    test_frame.feature_field = test_feature_field
    test_frame.raw_field = test_raw_field
    test_frame.populate_features()

    assert test_frame.get_feature(1).extreme == 10
    assert test_frame.get_feature(2).extreme == 20


def test_populate_features_invalid_negative_features():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_feature_field = test_field.copy()
    test_feature_field[3:5, 3:5] = -2
    test_feature_field[6:9, 6:9] = 2

    test_frame.feature_field = test_feature_field

    try:
        test_frame.populate_features()
    except NegativeIDError:
        pass


def test_populate_features_invalid_float_features():
    test_time = dt.datetime.now()
    test_frame = Frame()
    test_frame.time = test_time

    test_feature_field = test_field.copy()
    test_feature_field[3:5, 3:5] = 0.1
    test_feature_field[6:9, 6:9] = 2

    test_frame.feature_field = test_feature_field

    try:
        test_frame.populate_features()
    except FloatIDError:
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
    except FeaturesNotFoundError:
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
    except ArrayTypeError:
        pass


def test_get_next_available_feature_id_with_existing_max_id():
    test_frame = Frame()
    test_frame.max_id = 5
    assert test_frame.get_next_available_feature_id() == 6


def test_get_next_available_feature_id_with_no_existing_max_id_and_no_feature_field():
    test_frame = Frame()
    assert test_frame.get_next_available_feature_id() == 1


def test_get_next_available_feature_id_with_existing_field_and_no_existing_max_id():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_feature_field[2:4, 2:4] = 1
    test_feature_field[6:9, 6:9] = 2
    test_frame.feature_field = test_feature_field
    assert test_frame.get_next_available_feature_id() == 3


def test_promote_provisional_ids():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_feature_field[2:4, 2:4] = 1
    test_feature_field[6:9, 6:9] = 2
    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    # Set provisional ids for features
    feature1 = test_frame.get_feature(1)
    feature2 = test_frame.get_feature(2)
    feature1.provisional_id = 10
    feature2.provisional_id = 20

    # Promote provisional ids to main ids
    test_frame.promote_provisional_ids()

    assert feature1.id == 10
    assert feature2.id == 20
    assert feature1.provisional_id is None
    assert feature2.provisional_id is None

    # Now test that the features dict has also been updated
    expected_features_dict = {
        # Note this test has not updated the feature field itself, so the coords are
        # still based on the original feature ids
        10: Feature(10, np.where(test_feature_field == 1), test_frame.time),
        20: Feature(20, np.where(test_feature_field == 2), test_frame.time),
    }
    assert test_frame.get_features() == expected_features_dict


def test_update_fields_using_provisional_ids_with_valid_settings():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_feature_field[2:4, 2:4] = 1
    test_feature_field[6:9, 6:9] = 2
    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    # Set provisional ids for features
    feature1 = test_frame.get_feature(1)
    feature2 = test_frame.get_feature(2)
    feature1.provisional_id = 10
    feature2.provisional_id = 20

    # Also set the lifetime of these features
    feature1.lifetime = 5
    feature2.lifetime = 3

    # Update the feature field using the provisional ids
    test_frame.update_fields_using_provisional_ids()

    expected_feature_field = np.zeros((10, 10))
    expected_feature_field[2:4, 2:4] = 10
    expected_feature_field[6:9, 6:9] = 20

    expected_lifetime_field = np.zeros((10, 10))
    expected_lifetime_field[2:4, 2:4] = 5
    expected_lifetime_field[6:9, 6:9] = 3

    np.testing.assert_array_equal(
        test_frame.get_feature_field(), expected_feature_field
    )
    np.testing.assert_array_equal(
        test_frame.get_lifetime_field(), expected_lifetime_field
    )


def test_update_fields_using_provisional_ids_with_no_provisional_ids_set():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_feature_field[2:4, 2:4] = 1
    test_feature_field[6:9, 6:9] = 2
    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    # Update the feature field without setting any provisional ids
    test_frame.update_fields_using_provisional_ids()

    # The feature field should remain unchanged
    np.testing.assert_array_equal(test_frame.get_feature_field(), test_feature_field)


def test_update_fields_using_provisional_ids_with_no_feature_field():
    test_frame = Frame()
    test_frame.populate_features()  # No feature field set, so no features populated

    # Update the feature field without setting any provisional ids
    try:
        test_frame.update_fields_using_provisional_ids()
    except FeaturesNotFoundError:
        pass


def test_update_fields_using_provisional_ids_with_no_features():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_frame.feature_field = test_feature_field
    test_frame.populate_features()  # No features populated as feature field is all zeros

    # Update the feature field without any features
    try:
        test_frame.update_fields_using_provisional_ids()
    except FeaturesNotFoundError:
        pass


def test_get_new_features():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_feature_field[2:4, 2:4] = 1
    test_feature_field[6:8, 6:8] = 2
    test_feature_field[0:2, 0:2] = 3
    test_feature_field[8:10, 8:10] = 4
    test_frame.feature_field = test_feature_field
    test_frame.populate_features()
    # By definition, these features are new since they have not been tracked yet
    # and so will have lifetimes of 1 and will not have been found to split from
    # another storm

    # Set the lifetime of feature 3 to be 2 so that it is not considered new
    feature3 = test_frame.get_feature(3)
    feature3.lifetime = 2

    # Set the parent of feature 4 to be another feature so that it is not considered new
    feature4 = test_frame.get_feature(4)
    feature4.parent = 999  # Set to some arbitrary parent id

    new_features = test_frame.get_new_features()
    # Only expect feature 1 and 2 to be new
    expected_new_features = [test_frame.get_feature(1), test_frame.get_feature(2)]

    assert new_features == expected_new_features


def test_get_new_features_with_no_features():
    test_frame = Frame()
    new_features = test_frame.get_new_features()
    assert new_features == []


def test_get_dissipating_features():
    test_frame = Frame()
    test_feature_field = np.zeros((10, 10))
    test_feature_field[2:4, 2:4] = 1
    test_feature_field[6:8, 6:8] = 2
    test_feature_field[0:2, 0:2] = 3
    test_feature_field[8:10, 8:10] = 4
    test_frame.feature_field = test_feature_field
    test_frame.populate_features()

    # Set feature 1 as final timestep, and that it is not accreting
    # Should expect to see this feature in the list of dissipating features
    feature1 = test_frame.get_feature(1)
    feature1.set_as_final_timestep()  # Set as final timestep so that it can be considered dissipating

    # Set feature 2 as final timstep, but also set it to be accreted by another feature
    # Should not expect to see this feature in the list of dissipating features
    feature2 = test_frame.get_feature(2)
    feature2.set_as_final_timestep()  # Set as final timestep so that it can be considered dissipating
    # Set to some arbitrary id to indicate it is accreted by another feature
    feature2.accreted_in_next_frame_by = 999

    # Set the accreted_in_next_frame_by of feature 4 to be some value so that it is not considered dissipating
    feature4 = test_frame.get_feature(4)
    feature4.accreted_in_next_frame_by = 999  # Set to some arbitrary id

    dissipating_features = test_frame.get_dissipating_features()

    # Only expect feature 1 to be dissipating
    expected_dissipating_features = [test_frame.get_feature(1)]

    assert dissipating_features == expected_dissipating_features


def test_get_dissipating_features_with_no_features():
    test_frame = Frame()
    dissipating_features = test_frame.get_dissipating_features()
    assert dissipating_features == []


# TODO: add tests for get fields if this remains in the Frame class


def test_add_to_timeline_with_valid_frame():
    test_frame = Frame()
    test_time = dt.datetime.now()
    test_frame.time = test_time
    test_timeline = Timeline()
    test_timeline.add_to_timelime(test_frame)
    assert test_timeline.get_frame(test_time) == test_frame


def test_add_to_timeline_with_invalid_frame():
    test_timeline = Timeline()
    try:
        test_timeline.add_to_timelime("not a frame")
    except TypeError:
        pass


def test_add_to_timeline_with_frame_with_no_time():
    test_timeline = Timeline()
    test_frame = Frame()  # No time set for this frame
    try:
        test_timeline.add_to_timelime(test_frame)
    except ValueError:
        pass


def test_get_previous_frame_with_valid_time():
    test_timeline = Timeline()
    test_frame1 = Frame()
    test_frame2 = Frame()
    time1 = dt.datetime(2024, 1, 1, 0, 0, 0)
    time2 = dt.datetime(2024, 1, 1, 0, 5, 0)
    test_frame1.time = time1
    test_frame2.time = time2
    test_timeline.add_to_timelime(test_frame1)
    test_timeline.add_to_timelime(test_frame2)

    previous_frame = test_timeline.get_previous_frame(time2)
    assert previous_frame == test_frame1


def test_get_previous_frame_with_no_previous_frames():
    test_timeline = Timeline()
    test_frame = Frame()
    time = dt.datetime(2024, 1, 1, 0, 0, 0)
    test_frame.time = time
    test_timeline.add_to_timelime(test_frame)

    previous_frame = test_timeline.get_previous_frame(time)
    assert previous_frame is None


def test_get_previous_frame_with_empty_timeline():
    test_timeline = Timeline()
    time = dt.datetime(2024, 1, 1, 0, 0, 0)
    try:
        test_timeline.get_previous_frame(time)
    except ValueError:
        pass
