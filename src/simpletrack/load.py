import datetime as dt
import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from simpletrack.utils import check_arrays


class ConfigError(Exception):
    """
    Error thrown when one or more config input parameters are not valid
    """


class BaseLoader:
    """
    Base class for building custom loaders for use with Simple-Track.
    Provides base functionality for checking running iterator and checking loaded data
    for consistency and type before being passed to Simple-Track. Ensures the user only
    needs to worry about loading the data in the correct format.
    """

    def __init__(self, input_data: list[str] | dict) -> None:
        self.domain_shape = None
        self.input_data = input_data

    def __iter__(self):
        self.iter_idx = 0
        return self

    def user_definable_load(self, filename: str) -> list[dt.datetime, NDArray]:
        raise NotImplementedError

    def _check_loaded_data(
        self,
        output_time: dt.datetime,
        output_arr: NDArray,
    ) -> None:
        # Check consistency of data shape
        if self.domain_shape is None:
            self.domain_shape = output_arr.shape
            # Check that the domain sizes are even.
            # This requirement will ideally be relaxed in future
            # but currently required for flow_solver to work correctly.
            if not all([size % 2 == 0 for size in self.domain_shape]):
                msg = (
                    "Simple-Track requires even domain sizes, "
                    + f"input is shape {self.domain_shape}. Consider"
                    + " padding or cropping your data to meet this requirement."
                )
                raise ValueError(msg)
        output_arr = check_arrays(output_arr, shape=self.domain_shape, ndim=2)

        # Check output time is a sensible type
        if not isinstance(output_time, dt.datetime):
            raise TypeError(
                f"Expected 'output_time' to be datetime object, got {type(output_time)}"
            )

    def _validate_input_func(self, input_func: str | Callable) -> Callable:
        if callable(input_func):
            return input_func
        elif isinstance(input_func, str):
            return self._get_callable_from_str(input_func)
        else:
            raise TypeError(f"Expected str or callable, got {type(input_func)}")

    def _get_callable_from_str(self, func_str: str):
        file_path, function_name = func_str.split("|")

        # Load the module from the file path
        file_path = Path(file_path).resolve()  # Convert relative paths to absolute
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, function_name)


class FilenameIterator(BaseLoader):
    """
    Class used when user provides their own loading function, where each timestep is
    loaded as from a single file and iteration is performed over each filename.
    To use, provide the loader function and input data filenames to be iterated over
    by the loading function. The loading function must take a
    single input (filename) and should return a list of [datetime, array].
    The loader should be initialised with a list of filenames, which will be
    iterated through when the loader is used in Simple-Track.
    """

    def __init__(self, input_data, input_loader_func: str | Callable) -> None:
        super().__init__(input_data)
        # Check input is a list of filenames
        if not isinstance(input_data, (list, tuple)):
            raise TypeError(f"Expected input_data type list, got {type(input_data)}")

        self.user_definable_load = self._validate_input_func(input_loader_func)

    def __next__(self) -> list[dt.datetime, NDArray]:
        if self.iter_idx >= len(self.input_data):
            raise StopIteration
        next_fnm = self.input_data[self.iter_idx]
        self.iter_idx += 1
        time, data = self.user_definable_load(next_fnm)
        self._check_loaded_data(time, data)
        return time, data


class ArrayIterator(BaseLoader):
    """
    Class used when user provides their own loading function, where timesteps are
    loaded from one or more files and iteration is performed over a specified array
    dimension.
    To use, provide the loader function and input data filename(s) to be iterated over
    by the loading function, as well as the iterating dimension. The loading function
    must return two outputs: a list of datetime objects and an NDArray where the
    iterating dimension specified in the input is of the same size as the datetime list.
    This ensures each array slice matches with a corresponding datetime.
    """

    def __init__(
        self, input_data, input_loader_func: str | Callable, iterator_dim: int
    ) -> None:
        super().__init__(input_data)
        user_definable_load = self._validate_input_func(input_loader_func)
        self.all_times, all_data = user_definable_load(input_data)
        # Reshape array to make the iterating dimension the first dim
        # This will allow for more convenient array iteration
        self.iter_arr = np.moveaxis(all_data, iterator_dim, 0)

    def __next__(self) -> list[dt.datetime, NDArray]:
        if self.iter_idx >= len(self.input_data):
            raise StopIteration
        time = self.all_times[self.iter_idx]
        data = self.iter_arr[self.iter_idx]
        self._check_loaded_data(time, data)
        self.iter_idx += 1
        return time, data


class DictIterator(BaseLoader):
    """
    An alternative loading solution for users wish to load and/or pre-process their data
    elsewhere and pass it directly to Simple-Track. The input should be a dictionary
    with datetime keys and 2D array values. This will then iteratre through
    the dictionary in datetime order.
    """

    def __init__(self, input_dict: dict) -> None:
        self.domain_shape = None
        self.input_data = input_dict
        # Set the iterating list
        if not isinstance(input_dict, dict):
            raise TypeError(f"Expected input_data type dict, got {type(input_dict)}")
        self.iterator = sorted(input_dict.keys())
        if not all([isinstance(key, dt.datetime) for key in self.iterator]):
            raise TypeError("Expected all input keys to be of type dt.datetime")

    def __next__(self) -> list[NDArray, dt.datetime]:
        if self.iter_idx >= len(self.iterator):
            raise StopIteration
        time = self.iterator[self.iter_idx]
        data = self.input_data[time]
        self.iter_idx += 1
        self._check_loaded_data(time, data)
        return time, data


class LoadingBar:
    """
    Class for displaying a loading bar in the terminal. Initialised with the total
    number of items to load and the length of the loading bar, The
    "update_progress" method is then called to update the current progress
    """

    def __init__(self, total, bar_length=20):
        self.total = total
        self.bar_length = bar_length
        init_padding = int(self.bar_length) * " "
        print(f"Simple-Track Progress: [{init_padding}] 0/{self.total} (0%)", end="\r")

    def update_progress(self, current):
        fraction = current / self.total
        arrow = int(fraction * self.bar_length - 1) * "-" + ">"
        padding = int(self.bar_length - len(arrow)) * " "
        ending = "\n" if current == self.total else "\r"
        print(
            f"Simple-Track Progress: [{arrow}{padding}] {current}/{self.total} ({int(fraction * 100)}%) ",
            end=ending,
        )
