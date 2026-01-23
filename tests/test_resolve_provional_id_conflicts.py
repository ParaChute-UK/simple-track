import sys
import datetime as dt

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from FrameTracker import FrameTracker
from Feature import Feature
from Frame import Frame

import numpy as np


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

    print(test_frame.get_feature(2).accreted)
    print(test_frame.get_feature(3).accreted)

    # Check that this correctly removed the accreted id for Feature 2
    assert test_frame.get_feature(2).accreted is None

    # Check that this correctly kept the accreted id for Feature 2
    assert test_frame.get_feature(3).accreted == [10]
