import datetime as dt

import numpy as np
import pytest

from simpletrack.frame_output import FrameOutputManager, LoadOutput


@pytest.fixture()
def output_mwe_timeline(mwe_timeline, tmp_path):
    frame_output = FrameOutputManager(
        output_path=tmp_path,
        expt_name="mwe_test",
        start_time="2024-01-01 00:00:00",
        config_path=None,
        output_raw_data=True,
    )
    for frame in mwe_timeline.timeline.values():
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    return mwe_timeline, tmp_path


def test_output_mwe_timeline(output_mwe_timeline):
    """
    Test that the output files were created for each frame in the timeline
    """
    mwe_timeline, output_path = output_mwe_timeline
    strftime_pattern = "%Y%m%d_%H%M"
    missing_feature_idxs = [9, 10]
    missing_flow_idxs = [0, 8, 9, 10]

    for mwe_idx in range(10):
        frame_time_str = (
            dt.datetime(2024, 1, 1, 0, 0, 0) + dt.timedelta(minutes=5 * mwe_idx)
        ).strftime(strftime_pattern)

        assert (output_path / f"features_{frame_time_str}.field").is_file()
        assert (output_path / f"lifetime_{frame_time_str}.field").is_file()
        assert (output_path / f"raw_{frame_time_str}.field").is_file()

        # No flow fields expected for first frame, or frame without features (8, 9, 10)
        if mwe_idx not in missing_flow_idxs:
            assert (output_path / f"y-flow_{frame_time_str}.field").is_file()
            assert (output_path / f"x-flow_{frame_time_str}.field").is_file()

        # No features expected for empty frames 9 and 10
        if mwe_idx not in missing_feature_idxs:
            assert (output_path / f"frame_{frame_time_str}.txt").is_file()
            assert (output_path / f"frame_{frame_time_str}.csv").is_file()


def test_load_output_mwe_timeline(output_mwe_timeline):
    """
    Test that the output files can be loaded back into a timeline
    and that the features and flow match the original timeline
    """
    mwe_timeline, output_path = output_mwe_timeline

    load_output = LoadOutput(output_path)
    loaded_timeline = load_output.load_to_timeline()

    properties_to_check = [
        "id",
        "lifetime",
        "parent",
        "children",
        "centroid",
        "max",
        "mean",
        "dydx",
    ]

    for frame_time, frame in mwe_timeline.timeline.items():
        loaded_frame = loaded_timeline.get_frame(frame_time)

        # Compare fields (assert same precision as output)
        np.testing.assert_array_almost_equal(
            frame.feature_field, loaded_frame.feature_field, decimal=6
        )
        np.testing.assert_array_almost_equal(
            frame.lifetime_field, loaded_frame.lifetime_field, decimal=4
        )
        np.testing.assert_array_almost_equal(
            frame.raw_field, loaded_frame.raw_field, decimal=6
        )

        # Compare flow fields (only to 2dp precision, since this is a derived parameter
        # that is dependent on the flow scheme used)
        y_flow, x_flow = frame.get_flow()
        loaded_y_flow, loaded_x_flow = loaded_frame.get_flow()
        if y_flow is not None and loaded_y_flow is not None:
            np.testing.assert_array_almost_equal(y_flow, loaded_y_flow, decimal=2)
        if x_flow is not None and loaded_x_flow is not None:
            np.testing.assert_array_almost_equal(x_flow, loaded_x_flow, decimal=2)

        # Compare features
        assert len(frame.features) == len(loaded_frame.features)

        for feature_id, feature in frame.features.items():
            loaded_feature = loaded_frame.get_feature(feature_id)
            for prop in properties_to_check:
                assert getattr(feature, prop) == getattr(loaded_feature, prop)
            np.testing.assert_array_equal(feature.get_size(), loaded_feature.get_size())
