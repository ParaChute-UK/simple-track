import sys
from pathlib import Path
import pytest
import numpy as np
import datetime as dt

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from simple_track import SimpleTrack
from load import BaseLoader
from frame import Frame

test_time = dt.datetime(2026, 1, 1, 0, 0, 0)


def test_check_valid_config():
    test_config = {
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        "FEATURE": {"threshold": 4},
    }
    SimpleTrack._check_config(None, test_config)


def test_catch_missing_sections_in_config():
    test_config = {
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        # "FEATURE": {"threshold": 4},
    }
    try:
        SimpleTrack._check_config(None, test_config)
    except Exception:
        pass

    test_config = {
        # "PATH": {
        #     "data": "path",
        #     "loader": "loader",
        # },
        "FEATURE": {"threshold": 4},
    }
    try:
        SimpleTrack._check_config(None, test_config)
    except Exception:
        pass


def test_catch_missing_keys_in_config():
    test_config = {
        "PATH": {
            # "data": "path",
            "loader": "loader",
        },
        "FEATURE": {"threshold": 4},
    }
    try:
        SimpleTrack._check_config(None, test_config)
    except Exception:
        pass

    test_config = {
        "PATH": {
            "data": "path",
            # "loader": "loader",
        },
        "FEATURE": {"threshold": 4},
    }
    try:
        SimpleTrack._check_config(None, test_config)
    except Exception:
        pass

    test_config = {
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        # "FEATURE": {"threshold": 4},
    }
    try:
        SimpleTrack._check_config(None, test_config)
    except Exception:
        pass


@pytest.mark.parametrize(
    "extensions, expected_result",
    [
        [["nc", "nc"], True],
        [["npy", "npy"], True],
        [["nc", "npy"], True],  # TODO: this should probably raise something
        [["npy", "png"], True],
        [["pdf", "nc"], True],
        [["png", "png"], Exception],
    ],
)
def test_get_filenames_from_input_path(tmp_path, extensions, expected_result):
    # Create two files in temp directory
    files = [Path(f"{tmp_path}/f{f}.{ext}") for f, ext in enumerate(extensions)]
    files[0].parent.mkdir(exist_ok=True)
    for file in files:
        file.touch()
    expected_files = [f for f in files if f.suffix in [".nc", ".npy"]]

    try:
        retrieved_files = SimpleTrack._get_filenames_from_input_path(None, tmp_path)
        assert expected_files == retrieved_files
    except expected_result as e:
        print(e)


input_tests = (
    [
        [np.zeros((5, 5)), test_time, ValueError],
        ["Not an array", test_time, TypeError],
        [np.zeros((10, 10)), "Not a time", TypeError],
        [np.zeros((10, 10, 10)), test_time, ValueError],
        [np.zeros((10, 10)), test_time, True],
    ],
)


@pytest.mark.parametrize("test_arr, test_time, expected_result", *input_tests)
def test_BaseLoader(test_arr, test_time, expected_result):
    loader = BaseLoader()
    loader.domain_shape = (10, 10)
    try:
        loader._check_outputs(test_arr, test_time)
    except expected_result:
        pass


@pytest.mark.parametrize("test_arr, test_time, expected_result", *input_tests)
def test_Frame_import_data_and_time(test_arr, test_time, expected_result):
    frame = Frame()
    try:
        frame.import_data_and_time(test_arr, test_time)
    except expected_result:
        pass
