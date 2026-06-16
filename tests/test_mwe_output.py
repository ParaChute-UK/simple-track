import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from simpletrack.feature import Feature
from simpletrack.frame import Timeline
from simpletrack.track import Tracker


def generate_mwe_files(save_path=None):
    # Setup initial timestep with a single square cell
    mwe_domain = np.zeros((100, 100))

    mwe_dt1 = mwe_domain.copy()
    mwe_dt1[10:30, 10:30] = 1

    # Second timestep: advection of initial cell
    mwe_dt2 = mwe_domain.copy()
    mwe_dt2[15:35, 10:30] = 1

    # Third timestep: advection of initial cell, and creation of a new cell
    mwe_dt3 = mwe_domain.copy()
    mwe_dt3[20:40, 10:30] = 1
    # New cell created to the right
    mwe_dt3[15:35, 50:70] = 1

    # Fourth timestep: dissipation of inital cell, new cell advects
    mwe_dt4 = mwe_domain.copy()
    mwe_dt4[20:40, 50:70] = 1

    # Fifth timestep: new cell advects
    mwe_dt5 = mwe_domain.copy()
    mwe_dt5[25:45, 50:70] = 1

    # Sixth timestep: cell splitting
    mwe_dt6 = mwe_domain.copy()
    # Cell splits into two
    mwe_dt6[30:50, 48:58] = 1
    mwe_dt6[30:50, 62:72] = 1

    # Seventh timestep: advcetion merges cells
    mwe_dt7 = mwe_domain.copy()
    # Cells merge
    mwe_dt7[30:55, 50:70] = 1

    # Eigth timestep: advcetion
    mwe_dt8 = mwe_domain.copy()
    # Cells merge
    mwe_dt8[35:60, 50:70] = 1

    # Ninth timestep: no features
    mwe_dt9 = mwe_domain.copy()

    # Tenth timestep: tracking when both frames contain no features
    mwe_dt10 = mwe_domain.copy()

    mwe_fields = [
        mwe_dt1,
        mwe_dt2,
        mwe_dt3,
        mwe_dt4,
        mwe_dt5,
        mwe_dt6,
        mwe_dt7,
        mwe_dt8,
        mwe_dt9,
        mwe_dt10,
    ]
    if save_path is not None:
        # Make containing directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        for mwe_idx, mwe in enumerate(mwe_fields):
            np.savetxt(f"{save_path}/mwe_dt{mwe_idx + 1}.field", mwe)
    return mwe_fields


# pytest fixture with scope "session" means this setup will only run once, with the output used by
# any test that includes "mwe_timeline" as arg input
@pytest.fixture(scope="session")
def mwe_timeline() -> Timeline:
    mwe_fields = generate_mwe_files()

    mwe_config = {
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.3,
            "subdomain_size": 20,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }

    # Construct dict for passing to SimpleTrack
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_dict = {
        base_time + dt.timedelta(minutes=5 * int(mwe_idx)): mwe_data
        for mwe_idx, mwe_data in enumerate(mwe_fields)
    }

    timeline = Tracker(mwe_config).run(mwe_dict)
    return timeline


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


if __name__ == "__main__":
    mwe_file_path = "./mwe_test_files"
    Path(mwe_file_path).mkdir(parents=True, exist_ok=True)
    generate_mwe_files("./mwe_test_files")
