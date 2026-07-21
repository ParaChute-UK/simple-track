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
