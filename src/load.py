from numpy.typing import NDArray
import datetime as dt
from utils import check_arrays


def get_loader(loader_key: str):
    available_loaders = {
        "MWELoader": MWELoader,
        "CSETIndiaLoader": CSETIndiaLoader,
        "ChilboltonLoader": ChilboltonLoader,
    }
    loader = available_loaders[loader_key]()
    if not issubclass(type(loader), BaseLoader):
        print(type(loader))
        raise TypeError("Requested loader is not type LoadingManager")
    return loader


class BaseLoader:
    def __init__(self):
        self.domain_shape = None

    def _load(self, filename: str) -> list[NDArray, dt.datetime]:
        data, time = self.user_definable_load(filename)
        self._check_outputs(data, time)
        return data, time

    # TODO: rename this to something better
    def user_definable_load(self, filename: str) -> list[NDArray, dt.datetime]:
        raise NotImplementedError

    def _check_outputs(self, output_arr: NDArray, output_time: dt.datetime) -> None:
        # Check consistency of data shape
        if self.domain_shape is None:
            self.domain_shape = output_arr.shape
        output_arr = check_arrays(output_arr, shape=self.domain_shape, ndim=2)

        # Check output time is a sensible type
        if not isinstance(output_time, dt.datetime):
            raise TypeError(
                f"Expected 'output_time' to be datetime objcet, got {type(output_time)}"
            )


class MWELoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def user_definable_load(self, filename):
        import numpy as np

        base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
        data = np.load(filename)
        self.file_id = str(filename)
        mwe_idx = str(filename)[-5]
        time = base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        return data, time


class ChilboltonLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def user_definable_load(self, filename):
        import numpy as np
        from netCDF4 import Dataset as ncfile

        nc = ncfile(filename)
        data = nc.variables["var"][200:600, 250:550] / 32
        data = np.flipud(np.transpose(data))
        file_id = str(filename)[-9:-5]
        time = dt.time(hour=int(file_id[0:2]), minute=int(file_id[2:4]))
        return data, time


class CSETIndiaLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def user_definable_load(self, filename):
        import iris

        cube = iris.load_cube(filename, "precipitation_flux")
        # data is (401, 401) - cut the first elements out
        self.raw_field = cube.data[1:, 1:]
        assert self.raw_field.ndim == 2
        tcoord = cube.coord("time")
        time_points = tcoord.units.num2pydate(tcoord.points)
        assert len(time_points) == 1
        self.time = time_points[0]


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
