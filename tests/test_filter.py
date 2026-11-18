import numpy as np
import pytest

from simpletrack.filter import BaseFilter, InitFilter
from simpletrack.frame import Timeline


def test_copy_features_to_new_timeline(mwe_timeline):
    # Select a subset of features to copy
    features_to_copy = [1]

    filtered_timeline = BaseFilter()._copy_features_to_new_timeline(
        mwe_timeline, features_to_copy
    )

    # Check that the filtered timeline only contains selected feature in the features dict
    # and the numpy arrays
    for frame_time, frame in filtered_timeline.timeline.items():
        # Features dict should either be empty or contain the selected feature
        expected_features = [set(features_to_copy), set()]
        assert set(list(frame.features.keys())) in expected_features

        mwe_features = mwe_timeline.get_frame(frame_time).feature_field
        feature_mask = np.isin(mwe_features, features_to_copy)

        expected_raw_field = np.zeros_like(mwe_timeline.get_frame(frame_time).raw_field)
        expected_raw_field[feature_mask] = mwe_timeline.get_frame(frame_time).raw_field[
            feature_mask
        ]

        expected_lifetime_field = np.zeros_like(
            mwe_timeline.get_frame(frame_time).lifetime_field
        )
        expected_lifetime_field[feature_mask] = mwe_timeline.get_frame(
            frame_time
        ).lifetime_field[feature_mask]

        expected_feature_field = np.zeros_like(
            mwe_timeline.get_frame(frame_time).feature_field
        )
        expected_feature_field[feature_mask] = mwe_timeline.get_frame(
            frame_time
        ).feature_field[feature_mask]

        assert np.array_equal(frame.raw_field, expected_raw_field)
        assert np.array_equal(frame.lifetime_field, expected_lifetime_field)
        assert np.array_equal(frame.feature_field, expected_feature_field)


def test_copy_features_to_new_timeline_empty():
    # Create an empty timeline
    empty_timeline = Timeline()

    # Attempt to copy features from the empty timeline
    filtered_timeline = BaseFilter()._copy_features_to_new_timeline(
        empty_timeline, features_to_copy=[1]
    )

    # The filtered timeline should also be empty
    assert len(filtered_timeline.timeline) == 0


def test_copy_features_to_new_timeline_no_matching_features(mwe_timeline):
    # Select a feature that does not exist in the timeline
    features_to_copy = [999]

    filtered_timeline = BaseFilter()._copy_features_to_new_timeline(
        mwe_timeline, features_to_copy
    )

    # Check that the filtered timeline contains no features in the features dict
    for frame_time, frame in filtered_timeline.timeline.items():
        assert len(frame.features) == 0

        # The numpy arrays should be all zeros since no features were copied
        assert np.all(frame.raw_field == 0)
        assert np.all(frame.lifetime_field == 0)
        assert np.all(frame.feature_field == 0)


@pytest.fixture(scope="session")
def init_timeline_no_boundary(mwe_timeline):
    init_filter = InitFilter(boundary_buffer=None)
    init_timeline = init_filter.apply(mwe_timeline)
    return init_timeline


@pytest.fixture(scope="session")
def init_timeline_with_boundary(mwe_timeline):
    init_filter = InitFilter(boundary_buffer=20)
    init_timeline = init_filter.apply(mwe_timeline)
    return init_timeline


def test_init_filter_no_boundary_has_correct_features(init_timeline_no_boundary):
    # Check that only features 2 and 3 are present in the filtered timeline
    # since feature 1 is initialised in the first frame and should be filtered out
    for frame_time, frame in init_timeline_no_boundary.timeline.items():
        features = list(frame.features.keys())
        if len(features) > 0:
            assert set(features).issubset({2, 3})


def test_init_filter_no_boundary_has_correct_fields(init_timeline_no_boundary):
    for frame_time, frame in init_timeline_no_boundary.timeline.items():
        if len(frame.features) > 0:
            # Check that the feature_field only contains features 2 and 3
            assert np.all(np.isin(frame.feature_field, [0, 2, 3]))


def test_init_filter_with_boundary_has_correct_features(init_timeline_with_boundary):
    # Check that only feature 3 is present in the filtered timeline
    # since feature 1 is initialised in the first frame and feature 2 is too close to the boundary
    for frame_time, frame in init_timeline_with_boundary.timeline.items():
        features = list(frame.features.keys())
        if len(features) > 0:
            assert set(features).issubset({3})


def test_init_filter_with_boundary_has_correct_fields(init_timeline_with_boundary):
    for frame_time, frame in init_timeline_with_boundary.timeline.items():
        if len(frame.features) > 0:
            # Check that the feature_field only contains feature 3
            assert np.all(np.isin(frame.feature_field, [0, 3]))
