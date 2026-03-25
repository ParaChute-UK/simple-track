import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from simpletrack.frame import Frame
from simpletrack.load import BaseLoader, ConfigError, DictIterator
from simpletrack.track import Tracker
from simpletrack.utils import ArrayShapeError, ArrayTypeError

test_time = dt.datetime(2026, 1, 1, 0, 0, 0)


def test_check_valid_config():
    test_config = {
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        "FEATURE": {"threshold": 4},
    }
    Tracker._check_config(None, test_config)


def test_catch_missing_sections_in_config():
    test_config = {
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        # "FEATURE": {"threshold": 4},
    }
    try:
        Tracker._check_config(None, test_config)
    except ConfigError:
        pass

    test_config = {
        # "PATH": {
        #     "data": "path",
        #     "loader": "loader",
        # },
        "FEATURE": {"threshold": 4},
    }
    try:
        Tracker._check_config(None, test_config)
    except ConfigError:
        pass


def test_catch_missing_keys_in_config():
    test_config = {
        "PATH": {
            # "data": "path",
        },
        "FEATURE": {"threshold": 4},
    }
    try:
        Tracker._check_config(None, test_config)
    except ConfigError:
        pass

    test_config = {
        "PATH": {
            "data": "path",
        },
        "FEATURE": {},
    }
    try:
        Tracker._check_config(None, test_config)
    except ConfigError:
        pass


def test_invalid_config_input():
    try:
        Tracker(54)
    except TypeError:
        pass

    try:
        Tracker([4, 5, 6])
    except TypeError:
        pass


@pytest.mark.parametrize(
    "extensions, expected_result",
    [
        [["nc", "nc"], True],
        [["npy", "npy"], FileNotFoundError],
        [["nc", "npy"], True],
        [["npy", "png"], FileNotFoundError],
        [["pdf", "nc"], True],
        [["png", "png"], FileNotFoundError],
    ],
)
def test_get_filenames_from_input_path(tmp_path, extensions, expected_result):
    # Create two files in temp directory
    tmp_path.mkdir(exist_ok=True)
    files = [Path(f"{tmp_path}/f{f}.{ext}") for f, ext in enumerate(extensions)]
    for file in files:
        file.touch()
    expected_files = [f for f in files if f.suffix in [".nc"]]

    try:
        retrieved_files = Tracker.get_filenames_from_input_path(None, tmp_path)
        assert expected_files == retrieved_files
    except expected_result as e:
        print(e)


@pytest.mark.parametrize(
    "extensions_to_seed, extensions_to_find, expected_result",
    [
        [["field", "data"], [".field", ".data"], True],
        [["field"], ".field", True],
        [[".notdata"], [".field", ".data"], FileNotFoundError],
        [["field", "data"], [".nc", 1, 2], TypeError],
        [["field", "data"], 22, TypeError],
    ],
)
def test_get_filenames_from_input_path_with_added_valid_inputs(
    tmp_path, extensions_to_seed, extensions_to_find, expected_result
):
    # Create two files in temp directory
    tmp_path.mkdir(exist_ok=True)
    if isinstance(extensions_to_find, (list, str)):
        files = [
            Path(f"{tmp_path}/f{f}.{ext}") for f, ext in enumerate(extensions_to_seed)
        ]
    else:
        files = []

    for file in files:
        file.touch()
    expected_files = [f for f in files if f.suffix in extensions_to_find]

    try:
        retrieved_files = Tracker.get_filenames_from_input_path(
            None, tmp_path, file_type=extensions_to_find
        )
        assert expected_files == retrieved_files
    except expected_result as e:
        print(e)


input_tests = (
    [
        [np.zeros((5, 5)), test_time, ArrayShapeError],
        ["Not an array", test_time, ArrayTypeError],
        [np.zeros((10, 10)), "Not a time", TypeError],
        [np.zeros((10, 10, 10)), test_time, ArrayShapeError],
        [np.zeros((10, 10)), test_time, True],
    ],
)


@pytest.mark.parametrize("test_arr, test_time, expected_result", *input_tests)
def test_BaseLoader(test_arr, test_time, expected_result):
    # Pass empty list for the purposes of these tests
    loader = BaseLoader([])
    loader.domain_shape = (10, 10)
    try:
        loader._check_loaded_data(test_time, test_arr)
    except expected_result:
        pass


@pytest.mark.parametrize("test_arr, test_time, expected_result", *input_tests)
def test_Frame_import_time_and_data(test_arr, test_time, expected_result):
    frame = Frame()
    try:
        frame.import_time_and_data(test_time, test_arr)
    except expected_result:
        pass


@pytest.mark.parametrize(
    "invalid_input",
    [
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        np.array((1, 2, 3)),
        3,
        4.5,
        {"not a datetime": np.zeros((10, 10))},
        {dt.datetime.now: "not an array"},
    ],
)
def test_catch_invalid_Tracker_run_inputs(invalid_input):
    test_config = {
        "DATETIME": {"start_time": dt.datetime.now()},
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        "FEATURE": {"threshold": 4},
    }
    try:
        tracker = Tracker(test_config)
        tracker.run(invalid_input)
    except TypeError:
        pass


@pytest.mark.parametrize(
    "invalid_input", [[1, 2, 3], [1.0, 2.0, 3.0], np.array((1, 2, 3)), 3, 4.5]
)
def test_catch_invalid_Tracker_run_dict_inputs(invalid_input):
    test_config = {
        "DATETIME": {"start_time": dt.datetime.now()},
        "PATH": {
            "data": "path",
            "loader": "loader",
        },
        "FEATURE": {"threshold": 4},
    }
    try:
        tracker = Tracker(test_config)
        tracker.run(invalid_input)
    except TypeError:
        pass


def test_DictIterator_keys_properly_ordered():
    minutes = [48, 55, 20]
    test_times = [
        dt.datetime(year=2026, month=3, day=6, hour=14, minute=min) for min in minutes
    ]
    ordered_times = [test_times[2], test_times[0], test_times[1]]
    test_arr = np.zeros((10, 10))
    test_dict = {time: test_arr for time in test_times}

    test_iter = DictIterator(test_dict)
    for expected_time, (iter_time, __) in zip(ordered_times, test_iter):
        assert expected_time == iter_time
