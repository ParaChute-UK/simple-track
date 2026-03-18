import datetime as dt
from typing import Union

from numpy.typing import NDArray

from utils import check_arrays


class ConfigError(Exception):
    """
    Error thrown when one or more config input parameters are not valid
    """

    def __init__(self, message):
        super().__init__(message)


def get_loader(loader_key: str):
    available_loaders = {
        "MWELoader": MWELoader,
        "CSETIndiaLoader": CSETIndiaLoader,
        "ChilboltonLoader": ChilboltonLoader,
    }
    try:
        loader = available_loaders[loader_key]
    except KeyError:
        raise KeyError(f"Unknown loader: {loader_key}")
    if not issubclass(loader, BaseLoader):
        raise TypeError(f"Requested loader ({loader}) is not type BaseLoader")
    return loader


class BaseLoader:
    def __init__(self, input_data: Union[list[str] | dict]) -> None:
        self.domain_shape = None
        self.input_data = input_data
        # Set the iterating list
        if not isinstance(input_data, (list, tuple)):
            raise TypeError(f"Expected input_data type list, got {type(input_data)}")

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self) -> list[NDArray, dt.datetime]:
        if self.iter_idx >= len(self.input_data):
            raise StopIteration
        next_fnm = self.input_data[self.iter_idx]
        self.iter_idx += 1
        time, data = self.user_definable_load(next_fnm)
        self._check_loaded_data(time, data)
        return time, data

    # TODO: rename this to something better
    def user_definable_load(self, filename: str) -> list[NDArray, dt.datetime]:
        raise NotImplementedError

    def _check_loaded_data(
        self,
        output_time: dt.datetime,
        output_arr: NDArray,
    ) -> None:
        # Check consistency of data shape
        if self.domain_shape is None:
            self.domain_shape = output_arr.shape
        output_arr = check_arrays(output_arr, shape=self.domain_shape, ndim=2)

        # Check output time is a sensible type
        if not isinstance(output_time, dt.datetime):
            raise TypeError(
                f"Expected 'output_time' to be datetime object, got {type(output_time)}"
            )


class DictIterator(BaseLoader):
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


class MWELoader(BaseLoader):
    def __init__(self, filenames: list):
        super().__init__(filenames)

    def user_definable_load(self, filename):
        import numpy as np

        base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
        data = np.load(filename)
        self.file_id = str(filename)
        mwe_idx = str(filename)[-5]
        time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        return time, data


class ChilboltonLoader(BaseLoader):
    def __init__(self, filenames: list):
        super().__init__(filenames)

    def user_definable_load(self, filename):
        import numpy as np
        from netCDF4 import Dataset as ncfile

        nc = ncfile(filename)
        data = nc.variables["var"][200:600, 250:550] / 32
        data = np.flipud(np.transpose(data))
        date_id = str(filename)[-18:-11]
        time_id = str(filename)[-9:-5]
        time = dt.datetime(
            year=int(date_id[0:4]),
            month=int(date_id[4:6]),
            day=int(date_id[6:]),
            hour=int(time_id[0:2]),
            minute=int(time_id[2:4]),
        )
        return time, data


class CSETIndiaLoader(BaseLoader):
    def __init__(self, filenames):
        super().__init__(filenames)

    def user_definable_load(self, filename):
        import iris

        cube = iris.load_cube(filename, "precipitation_flux")
        # data is (401, 401) - cut the first elements out
        data = cube.data[1:, 1:]
        assert data.ndim == 2
        tcoord = cube.coord("time")
        time_points = tcoord.units.num2pydate(tcoord.points)
        assert len(time_points) == 1
        time_points = time_points[0]
        return time_points, data


class LoadingBar:
    def __init__(self, total, bar_length=20):
        self.total = total
        self.bar_length = bar_length
        init_padding = int(self.bar_length) * " "
        print(f"Progress: [{init_padding}] 0%", end="\r")

    def update_progress(self, current):
        fraction = current / self.total
        arrow = int(fraction * self.bar_length - 1) * "-" + ">"
        padding = int(self.bar_length - len(arrow)) * " "
        ending = "\n" if current == self.total else "\r"
        print(f"Progress: [{arrow}{padding}] {int(fraction * 100)}%", end=ending)
