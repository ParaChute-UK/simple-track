import datetime as dt

import numpy as np

from simpletrack import Tracker
from simpletrack.load import ArrayIterator, BaseLoader, FilenameIterator

from .conftest import generate_mwe_files


def test_mwe_filenameiter_loader_with_str_loader_func(tmp_path):
    # Create mwe test files in tmp_path and return to variable
    mwe_fields = generate_mwe_files(tmp_path)
    fnms = Tracker.get_filenames_from_input_path(None, f"{tmp_path}/*.field")

    # Set path for loader to find loading function
    func_path = "tests/mwe_loader.py|load_mwe"

    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    all_times = [
        base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        for mwe_idx in range(len(mwe_fields))
    ]

    # Should produce datetime, array iterable
    iterator = FilenameIterator(fnms, func_path)

    for iter_idx, iter in enumerate(iterator):
        assert iter[0] == all_times[iter_idx]
        np.testing.assert_array_equal(iter[1], mwe_fields[iter_idx])


def test_mwe_filenameiter_loader_with_callable_loader_func(tmp_path):
    # Create mwe test files in tmp_path and return to variable
    mwe_fields = generate_mwe_files(tmp_path)
    fnms = Tracker.get_filenames_from_input_path(None, f"{tmp_path}/*.field")

    # Set path for loader to find loading function
    func_path = "tests/mwe_loader.py|load_mwe"

    callable_func = BaseLoader._get_callable_from_str(None, func_path)

    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    all_times = [
        base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        for mwe_idx in range(len(mwe_fields))
    ]

    # Should produce datetime, array iterable
    iterator = FilenameIterator(fnms, callable_func)

    for iter_idx, iter in enumerate(iterator):
        assert iter[0] == all_times[iter_idx]
        np.testing.assert_array_equal(iter[1], mwe_fields[iter_idx])


def test_mwe_arrayiter_loader_with_str_loader_func(tmp_path):
    # Create mwe test files in tmp_path and return to variable
    mwe_fields = generate_mwe_files(tmp_path)
    fnms = Tracker.get_filenames_from_input_path(None, f"{tmp_path}/*.field")

    # Set path for loader to find loading function
    func_path = "tests/mwe_loader.py|load_all_mwe"

    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    all_times = [
        base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        for mwe_idx in range(len(mwe_fields))
    ]

    # Should produce datetime, array iterable
    iterator = ArrayIterator(fnms, func_path, iterator_dim=0)

    for iter_idx, iter in enumerate(iterator):
        assert iter[0] == all_times[iter_idx]
        np.testing.assert_array_equal(iter[1], mwe_fields[iter_idx])


def test_mwe_arrayiter_loader_with_callable_loader_func(tmp_path):
    # Create mwe test files in tmp_path and return to variable
    mwe_fields = generate_mwe_files(tmp_path)
    fnms = Tracker.get_filenames_from_input_path(None, f"{tmp_path}/*.field")

    # Set path for loader to find loading function
    func_path = "tests/mwe_loader.py|load_all_mwe"

    callable_func = BaseLoader._get_callable_from_str(None, func_path)

    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    all_times = [
        base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        for mwe_idx in range(len(mwe_fields))
    ]

    # Should produce datetime, array iterable
    iterator = ArrayIterator(fnms, callable_func, iterator_dim=0)

    for iter_idx, iter in enumerate(iterator):
        assert iter[0] == all_times[iter_idx]
        np.testing.assert_array_equal(iter[1], mwe_fields[iter_idx])
