import sys
from pathlib import Path
import pytest

sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track"
)
sys.path.append(
    "/Users/workcset/Library/CloudStorage/OneDrive-UniversityofReading/Code/simple-track/src"
)
from simple_track import SimpleTrack


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
