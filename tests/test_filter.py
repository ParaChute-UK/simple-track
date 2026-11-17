import numpy as np
import pytest

from simpletrack.filter import BaseFilter


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
