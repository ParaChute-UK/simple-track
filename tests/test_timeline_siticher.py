import datetime as dt

import numpy as np
import pytest

from simpletrack.frame_output import FrameOutputManager, LoadOutput
from simpletrack.timeline_stitcher import TimelineStitcher


@pytest.fixture()
def output_mwe_timeline_in_batches(mwe_timeline, tmp_path):
    # Split the timeline into three batches and output each batch separately
    mwe_batch_1 = [0, 1, 2]
    mwe_batch_2 = [3, 4, 5]
    mwe_batch_3 = [6, 7, 8, 9]

    timeline_keys = list(mwe_timeline.timeline.keys())

    # Output the first batch
    frame_output = FrameOutputManager(
        output_path=f"{tmp_path}/batch_1",
        expt_name="mwe_test",
        start_time="2024-01-01 00:00:00",
        config_path="./test_config.yaml",
        output_raw_data=True,
    )

    for mwe_idx in mwe_batch_1:
        frame = mwe_timeline.timeline[timeline_keys[mwe_idx]]
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    # Output the second batch
    frame_output = FrameOutputManager(
        output_path=f"{tmp_path}/batch_2",
        expt_name="mwe_test",
        start_time="2024-01-01 00:15:00",
        config_path="./test_config.yaml",
        output_raw_data=True,
    )
    for mwe_idx in mwe_batch_2:
        frame = mwe_timeline.timeline[timeline_keys[mwe_idx]]
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    # Output the third batch
    frame_output = FrameOutputManager(
        output_path=f"{tmp_path}/batch_3",
        expt_name="mwe_test",
        start_time="2024-01-01 00:30:00",
        config_path="./test_config.yaml",
        output_raw_data=True,
    )
    for mwe_idx in mwe_batch_3:
        frame = mwe_timeline.timeline[timeline_keys[mwe_idx]]
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    return tmp_path


@pytest.fixture()
def mwe_timeline_batches(output_mwe_timeline_in_batches):
    tmp_path = output_mwe_timeline_in_batches
    # Return all timeline batches after loading in from output
    batch_1 = LoadOutput(f"{tmp_path}/batch_1").load_to_timeline()
    batch_2 = LoadOutput(f"{tmp_path}/batch_2").load_to_timeline()
    batch_3 = LoadOutput(f"{tmp_path}/batch_3").load_to_timeline()

    return batch_1, batch_2, batch_3


def test_timeline_stitcher_from_timelines(mwe_timeline_batches):
    """
    Test that TimelineStitcher successfully merges multiple timelines
    in the correct order
    """
    batch_1, batch_2, batch_3 = mwe_timeline_batches

    # Stitch the batches together
    stitched_timeline = TimelineStitcher([batch_1, batch_2, batch_3]).run()

    # Check that the stitched timeline has the correct number of frames
    assert len(stitched_timeline.timeline) == 10

    # Check that the stitched timeline has the correct frame times
    expected_frame_times = [
        dt.datetime(2024, 1, 1, 0, 0, 0),
        dt.datetime(2024, 1, 1, 0, 5, 0),
        dt.datetime(2024, 1, 1, 0, 10, 0),
        dt.datetime(2024, 1, 1, 0, 15, 0),
        dt.datetime(2024, 1, 1, 0, 20, 0),
        dt.datetime(2024, 1, 1, 0, 25, 0),
        dt.datetime(2024, 1, 1, 0, 30, 0),
        dt.datetime(2024, 1, 1, 0, 35, 0),
        dt.datetime(2024, 1, 1, 0, 40, 0),
        dt.datetime(2024, 1, 1, 0, 45, 0),
    ]
    assert list(stitched_timeline.timeline.keys()) == expected_frame_times


def test_timeline_stitcher_from_str(output_mwe_timeline_in_batches):
    """
    Test that TimelineStitcher successfully merges multiple str path to output data
    in the correct order
    """
    tmp_path = output_mwe_timeline_in_batches
    batch_1 = f"{tmp_path}/batch_1"
    batch_2 = f"{tmp_path}/batch_2"
    batch_3 = f"{tmp_path}/batch_3"

    # Stitch the batches together
    stitched_timeline = TimelineStitcher([batch_1, batch_2, batch_3]).run()

    # Check that the stitched timeline has the correct number of frames
    assert len(stitched_timeline.timeline) == 10

    # Check that the stitched timeline has the correct frame times
    expected_frame_times = [
        dt.datetime(2024, 1, 1, 0, 0, 0),
        dt.datetime(2024, 1, 1, 0, 5, 0),
        dt.datetime(2024, 1, 1, 0, 10, 0),
        dt.datetime(2024, 1, 1, 0, 15, 0),
        dt.datetime(2024, 1, 1, 0, 20, 0),
        dt.datetime(2024, 1, 1, 0, 25, 0),
        dt.datetime(2024, 1, 1, 0, 30, 0),
        dt.datetime(2024, 1, 1, 0, 35, 0),
        dt.datetime(2024, 1, 1, 0, 40, 0),
        dt.datetime(2024, 1, 1, 0, 45, 0),
    ]
    assert list(stitched_timeline.timeline.keys()) == expected_frame_times


@pytest.fixture()
def output_mwe_timeline_in_batches_with_overlap(mwe_timeline, tmp_path):
    # Split the timeline into three batches and output each batch separately
    mwe_batch_1 = [0, 1, 2, 3]
    mwe_batch_2 = [3, 4, 5, 6]
    mwe_batch_3 = [6, 7, 8, 9]

    timeline_keys = list(mwe_timeline.timeline.keys())

    # Output the first batch
    frame_output = FrameOutputManager(
        output_path=f"{tmp_path}/batch_1",
        expt_name="mwe_test",
        start_time="2024-01-01 00:00:00",
        config_path="./test_config.yaml",
        output_raw_data=True,
    )

    for mwe_idx in mwe_batch_1:
        frame = mwe_timeline.timeline[timeline_keys[mwe_idx]]
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    # Output the second batch
    frame_output = FrameOutputManager(
        output_path=f"{tmp_path}/batch_2",
        expt_name="mwe_test",
        start_time="2024-01-01 00:20:00",
        config_path="./test_config.yaml",
        output_raw_data=True,
    )
    for mwe_idx in mwe_batch_2:
        frame = mwe_timeline.timeline[timeline_keys[mwe_idx]]
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    # Output the third batch
    frame_output = FrameOutputManager(
        output_path=f"{tmp_path}/batch_3",
        expt_name="mwe_test",
        start_time="2024-01-01 00:40:00",
        config_path="./test_config.yaml",
        output_raw_data=True,
    )
    for mwe_idx in mwe_batch_3:
        frame = mwe_timeline.timeline[timeline_keys[mwe_idx]]
        frame_output.features_to_txt(frame)
        frame_output.features_to_csv(frame)
        frame_output.fields_to_npy(frame)

    return tmp_path


@pytest.fixture()
def mwe_timeline_batches_with_overlap(output_mwe_timeline_in_batches_with_overlap):
    tmp_path = output_mwe_timeline_in_batches_with_overlap
    # Return all timeline batches after loading in from output
    batch_1 = LoadOutput(f"{tmp_path}/batch_1").load_to_timeline()
    batch_2 = LoadOutput(f"{tmp_path}/batch_2").load_to_timeline()
    batch_3 = LoadOutput(f"{tmp_path}/batch_3").load_to_timeline()

    return batch_1, batch_2, batch_3


def test_timeline_stitcher_does_not_repeat_inputs(mwe_timeline_batches_with_overlap):
    """
    Test that TimelineStitcher does not repeat frames when the same
    frame of data is provided multiple times in different timelines
    """

    batch_1, batch_2, batch_3 = mwe_timeline_batches_with_overlap

    # Stitch the batches together
    stitched_timeline = TimelineStitcher([batch_1, batch_2, batch_3]).run()

    # Check that the stitched timeline has the correct number of frames
    assert len(stitched_timeline.timeline) == 10

    # Check that the stitched timeline has the correct frame times
    expected_frame_times = [
        dt.datetime(2024, 1, 1, 0, 0, 0),
        dt.datetime(2024, 1, 1, 0, 5, 0),
        dt.datetime(2024, 1, 1, 0, 10, 0),
        dt.datetime(2024, 1, 1, 0, 15, 0),
        dt.datetime(2024, 1, 1, 0, 20, 0),
        dt.datetime(2024, 1, 1, 0, 25, 0),
        dt.datetime(2024, 1, 1, 0, 30, 0),
        dt.datetime(2024, 1, 1, 0, 35, 0),
        dt.datetime(2024, 1, 1, 0, 40, 0),
        dt.datetime(2024, 1, 1, 0, 45, 0),
    ]
    assert list(stitched_timeline.timeline.keys()) == expected_frame_times


# Further tests for TimelineStitcher are found in test_mwe_output using the
# "mwe_timeline_stitched" fixture
