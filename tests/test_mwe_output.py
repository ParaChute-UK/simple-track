import datetime as dt

import numpy as np

from simpletrack.feature import Feature
from simpletrack.track import Tracker


def test_first_mwe_outputs(mwe_timeline):
    """
    Test that a single feature exists in the first frame
    with no parent, children, and lifetime of 1, with expected
    size and centroid.

    Also test that there is no flow in the first frame,
    and that the feature is correctly identified as a new feature
    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    frame = mwe_timeline.get_frame(base_time)

    # test there is a single feature
    assert len(frame.features) == 1

    # test feature properties
    feature = frame.get_feature(1)
    assert isinstance(feature, Feature)
    assert feature.id == 1
    assert feature.lifetime == 1
    assert feature.parent is None
    assert feature.children is None
    assert feature.centroid == (19.5, 19.5)
    assert feature.get_size() == 400

    # Test there is no flow
    assert feature.dydx == ()
    assert frame.get_flow() == (None, None)


def test_second_mwe_outputs(mwe_timeline):
    """
    Test that there is still a single feature with the same id
    as the feature in the first frame and with an incremented lifetime
    Also test that there is a flow from the first frame to the second
    (although don't need to test exact flow values, just that they are nonzero)

    Also test that there is still no parent or children, and the centroid
    has been updated as expected

    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 1
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there is a single feature
    assert len(frame.features) == 1

    # test feature properties
    feature = frame.get_feature(1)
    assert isinstance(feature, Feature)
    assert feature.id == 1
    assert feature.lifetime == 2
    assert feature.parent is None
    assert feature.children is None
    assert feature.centroid == (24.5, 19.5)
    assert feature.get_size() == 400

    # Test there is a flow across the feature
    assert feature.dydx != ()
    assert np.all(frame.get_flow()) is not None
    # Full flow cannot be neatly anticipated with this MWE due to Fourier transformations of
    # fields/data with "sharp" edges (binary), so for this timestep, just test
    # that the maximum is within a reasonable (large) range
    max_yflow = np.max(frame.get_flow()[0])
    assert np.isclose(max_yflow, 5, atol=1)


def test_third_mwe_outputs(mwe_timeline):
    """
    Test that there are now two features, with the first having the same
    id as the feature in the second frame and an incremented lifetime, and
    the second feature having a new id and a lifetime of 1. Also test that
    this new feature is not a child of the first feature, and this it is
    correctly identified as a new feature
    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 2
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there are two features
    assert len(frame.features) == 2

    # test feature properties for feature 1
    feature = frame.get_feature(1)
    assert isinstance(feature, Feature)
    assert feature.id == 1
    assert feature.lifetime == 3
    assert feature.parent is None
    assert feature.children is None
    assert feature.centroid == (29.5, 19.5)
    assert feature.get_size() == 400

    # test feature properties for feature 2
    feature = frame.get_feature(2)
    assert isinstance(feature, Feature)
    assert feature.id == 2
    assert feature.lifetime == 1
    assert feature.parent is None
    assert feature.children is None
    assert feature.centroid == (24.5, 59.5)
    assert feature.get_size() == 400

    # Test there is a flow across the feature
    assert feature.dydx != ()
    assert np.all(frame.get_flow()) is not None
    # Full flow cannot be neatly anticipated with this MWE due to Fourier transformations of
    # fields/data with "sharp" edges (binary), so for this timestep, just test
    # that the maximum is within a reasonable (large) range
    max_yflow = np.max(frame.get_flow()[0])
    assert np.isclose(max_yflow, 5, atol=1)


def test_fourth_mwe_outputs(mwe_timeline):
    """
    Test that the first feature is no longer present, and that the second
    feature has the same id as in the previous frame, and with an incremented
    lifetime.

    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 3
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there we are back to one feature
    assert len(frame.features) == 1

    # test feature properties for feature 2
    feature = frame.get_feature(2)
    assert isinstance(feature, Feature)
    assert feature.id == 2
    assert feature.lifetime == 2
    assert feature.parent is None
    assert feature.children is None
    assert feature.centroid == (29.5, 59.5)
    assert feature.get_size() == 400

    # Test there is a flow across the feature
    assert feature.dydx != ()
    assert np.all(frame.get_flow()) is not None
    # Flow solver does not like this setup and does not produce reasonble
    # flow for the advected feature only. This will likely be improved
    # using an optical flow solver


def test_fifth_mwe_outputs(mwe_timeline):
    """
    Test that the second feature advects as expected
    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 4
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there we are back to one feature
    assert len(frame.features) == 1

    # test feature properties for feature 2
    feature = frame.get_feature(2)
    assert isinstance(feature, Feature)
    assert feature.id == 2
    assert feature.lifetime == 3
    assert feature.parent is None
    assert feature.children is None
    assert feature.centroid == (34.5, 59.5)
    assert feature.get_size() == 400

    # Test there is a flow across the feature
    assert feature.dydx != ()
    assert np.all(frame.get_flow()) is not None
    # Full flow cannot be neatly anticipated with this MWE due to Fourier transformations of
    # fields/data with "sharp" edges (binary), so for this timestep, just test
    # that the maximum is within a reasonable (large) range
    max_yflow = np.max(frame.get_flow()[0])
    assert np.isclose(max_yflow, 5, atol=1)


def test_sixth_mwe_outputs(mwe_timeline):
    """
    Test that the feature has split into two, with one feature retaining the
    previous id and an incremented lifetime, and the other feature having a new id
    and a retained lifetime of 4. Also test that the new feature is correctly identified as
    a child, and the old feature is a parent.
    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 5
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there we are back to one feature
    assert len(frame.features) == 2

    # test feature properties for feature 2
    feature = frame.get_feature(2)
    assert isinstance(feature, Feature)
    assert feature.id == 2
    assert feature.lifetime == 4
    assert feature.parent is None
    assert feature.children == [3]
    assert feature.centroid == (39.5, 52.5)
    assert feature.get_size() == 200

    # Test there is a flow across the feature
    assert feature.dydx != ()
    assert np.all(frame.get_flow()) is not None

    # test feature properties for feature 2
    feature = frame.get_feature(3)
    assert isinstance(feature, Feature)
    assert feature.id == 3
    assert feature.lifetime == 4  # retain_lifetime_on_split defaults to True
    assert feature.parent == 2
    assert feature.children is None
    assert feature.centroid == (39.5, 66.5)
    assert feature.get_size() == 200


def test_seventh_mwe_outputs(mwe_timeline):
    """
    Test that the two features have merged back into one, with the same id
    being retained from the older timestep and with an incremented lifetime.
    Also test that the merged feature has correctly identified the split feature
    as being accreted by the resulting feature.
    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 6
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there we are back to one feature
    assert len(frame.features) == 1

    # test feature properties for feature 2
    feature = frame.get_feature(2)
    assert isinstance(feature, Feature)
    assert feature.id == 2
    assert feature.lifetime == 5
    assert feature.parent is None
    assert feature.children is None
    assert feature.accreted == [3]
    assert feature.centroid == (42, 59.5)
    assert feature.get_size() == 500

    # Test there is a flow across the feature
    assert feature.dydx != ()
    assert np.all(frame.get_flow()) is not None


def test_ninth_mwe_outputs(mwe_timeline):
    """
    Test that there are no features in the ninth timestep
    """
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 8
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)

    # test there we are back to one feature
    assert len(frame.features) == 0


def test_split_merge_event_with_larger_split_feature_than_merging_feature():
    domain_shape = (100, 100)

    field0 = np.zeros(domain_shape)
    field0[20:80, 20:40] = 1
    field0[20:40, 40:60] = 1

    field1 = np.zeros(domain_shape)
    field1[20:80, 20:40] = 1
    field1[20:40, 40:60] = 1
    field1[45:65, 55:60] = 2

    field2 = np.zeros(domain_shape)
    field2[20:80, 20:40] = 1
    field2[20:45, 45:60] = 2
    field2[45:65, 55:60] = 2

    fields = [field0, field1, field2]

    # Setup tracking
    # Need to use an artificially small threshold to set up MWE, especially for case 2b.
    # Perhaps this indicates that case 2b isn't a realistic scenario, but ST should still work
    # regardless of the threshold
    config = {"FEATURE": {"threshold": 0.5}, "TRACKING": {"overlap_threshold": 0.1}}

    # Setup data
    time_base = dt.datetime.now()
    times = [time_base + dt.timedelta(hours=i) for i in range(len(fields))]
    data = {time: field for time, field in zip(times, fields)}

    timeline = Tracker(config).run(data)

    # Test the results in frame 3
    frame3 = timeline.get_frame(times[2])
    assert len(frame3.features) == 2

    # Should expect that this feature will be identified as a child of the parent feature
    # and should be given a new id of 3. It should have the same lifetime as the parent
    # since retain_lifetime_on_split defaults to True

    frame3_feature3 = frame3.get_feature(3)
    assert isinstance(frame3_feature3, Feature)
    assert frame3_feature3.id == 3
    assert frame3_feature3.lifetime == 3
    assert frame3_feature3.parent == 1


def test_split_merge_event_with_smaller_split_feature_than_merging_feature():
    domain_shape = (100, 100)

    field0 = np.zeros(domain_shape)
    field0[10:90, 5:40] = 1
    field0[20:40, 40:60] = 1

    field1 = np.zeros(domain_shape)
    field1[10:90, 5:40] = 1
    field1[20:40, 40:60] = 1
    field1[45:90, 55:80] = 2

    field2 = np.zeros(domain_shape)
    field2[10:90, 5:40] = 1
    field2[25:45, 45:70] = 2
    field2[45:90, 55:80] = 2

    fields = [field0, field1, field2]

    # Setup tracking
    # Need to use an artificially small threshold to set up MWE, especially for case 2b.
    # Perhaps this indicates that case 2b isn't a realistic scenario, but ST should still work
    # regardless of the threshold
    config = {"FEATURE": {"threshold": 0.5}, "TRACKING": {"overlap_threshold": 0.1}}

    # Setup data
    time_base = dt.datetime.now()
    times = [time_base + dt.timedelta(hours=i) for i in range(len(fields))]
    data = {time: field for time, field in zip(times, fields)}

    timeline = Tracker(config).run(data)

    # Test the results in frame 3
    frame3 = timeline.get_frame(times[2])
    assert len(frame3.features) == 2

    # Should expect that the split feature will merge with the existing feature
    # and retain the id of the merging feature (feature 2)
    frame3_feature2 = frame3.get_feature(2)
    assert isinstance(frame3_feature2, Feature)
    assert frame3_feature2.id == 2
    assert frame3_feature2.lifetime == 2
    assert frame3_feature2.parent is None


if __name__ == "__main__":
    from pathlib import Path

    from conftest import generate_mwe_files

    mwe_file_path = "./mwe_test_files"
    Path(mwe_file_path).mkdir(parents=True, exist_ok=True)
    generate_mwe_files("./mwe_test_files")
