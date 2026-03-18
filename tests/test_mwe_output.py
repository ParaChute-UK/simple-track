import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Documents/Code/simple-track/src"
)
from simple_track import SimpleTrack


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

    mwe_fields = [
        mwe_dt1,
        mwe_dt2,
        mwe_dt3,
        mwe_dt4,
        mwe_dt5,
        mwe_dt6,
        mwe_dt7,
        mwe_dt8,
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
def mwe_timeline():
    mwe_fields = generate_mwe_files()

    # Construct dict for passing to SimpleTrack
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_dict = {
        base_time + dt.timedelta(minutes=5 * int(mwe_idx)): mwe_data
        for mwe_idx, mwe_data in enumerate(mwe_fields)
    }

    mwe_config = {
        "DATETIME": {"start_time": "2012-08-25 14:05:00", "time_interval": 5},
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.2,  # Lower overlap needed for some reason
            "subdomain_size": 10,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }
    timeline = SimpleTrack(mwe_config).run(mwe_dict)
    return timeline


def test_first_mwe_outputs(mwe_timeline):
    # TODO
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    frame = mwe_timeline.get_frame(base_time)
    print(frame)


def test_second_mwe_outputs(mwe_timeline):
    # TODO
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    mwe_idx = 1
    frame_time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
    frame = mwe_timeline.get_frame(frame_time)
    print(frame)


if __name__ == "__main__":
    mwe_file_path = "./mwe_test_files"
    Path(mwe_file_path).mkdir(parents=True, exist_ok=True)
    generate_mwe_files("./mwe_test_files")
