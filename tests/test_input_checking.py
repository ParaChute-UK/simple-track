import sys

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
