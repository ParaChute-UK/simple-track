from numpy.typing import NDArray
import numpy as np


class Feature:
    def __init__(self, id: int, feature_coords: NDArray[np.integer] = None) -> None:
        if feature_coords.shape[0] != 2 or feature_coords.ndim != 2:
            exc_str = "Expected a 2D array with first dimension of size 2 "
            exc_str += f"got {feature_coords.ndim} array with first dimension"
            exc_str += f"of size {feature_coords.shape[0]}"
            raise TypeError(exc_str)

        self._id = id
        self._feature_coords = feature_coords
        self._centroid = None
        self._lifetime = 1
        self._accreted = None
        self._parent = None
        self._child = None
        self._dx = 0
        self._dy = 0

    @property
    def id(self) -> int:
        return self._id

    @property
    def centroid(self) -> tuple:
        if self._centroid is None:
            self._centroid = self.calculate_centroid()
        return self._centroid

    @property
    def coords(self) -> NDArray[np.integer]:
        return self._feature_coords

    @property
    def lifetime(self) -> int:
        return self._lifetime

    @property
    def accreted(self) -> list[int]:
        return self._accreted

    @property
    def parent(self) -> int:
        return self._parent

    @property
    def child(self) -> list[int]:
        return self._child

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dy(self) -> float:
        return self._dy

    @coords.setter
    def coords(self, new_coords: NDArray[np.integer]) -> None:
        self._feature_coords = new_coords
        self._centroid = self.calculate_centroid()  # Update centroid when coords change

    @parent.setter
    def parent(self, parent_id: int) -> None:
        self._parent = parent_id

    @dx.setter
    def dx(self, delta_x: float) -> None:
        self._dx = delta_x

    @dy.setter
    def dy(self, delta_y: float) -> None:
        self._dy = delta_y

    def increment_lifetime(self) -> None:
        self._lifetime += 1

    def calculate_centroid(self) -> tuple:
        y_centroid = np.mean(self._feature_coords[0, :])
        x_centroid = np.mean(self._feature_coords[1, :])
        return (y_centroid, x_centroid)

    def accretes(self, feature_ids: int | list[int]) -> None:
        if self._accreted is None:
            self._accreted = []
        if isinstance(feature_ids, int):
            self._accreted.append(feature_ids)
        else:
            self._accreted.extend(feature_ids)

    def spawns(self, feature_ids: int | list[int]) -> None:
        if self._child is None:
            self._child = []
        if isinstance(feature_ids, int):
            self._child.append(feature_ids)
        else:
            self._child.extend(feature_ids)

    def get_x_centroid(self) -> float:
        return self.centroid[1]

    def get_y_centroid(self) -> float:
        return self.centroid[0]

    def get_x_min(self) -> int:
        return np.min(self._feature_coords[1, :])

    def get_x_max(self) -> int:
        return np.max(self._feature_coords[1, :])

    def get_y_min(self) -> int:
        return np.min(self._feature_coords[0, :])

    def get_y_max(self) -> int:
        return np.max(self._feature_coords[0, :])

    def get_x_extent(self) -> int:
        return self.get_x_max() - self.get_x_min()

    def get_y_extent(self) -> int:
        return self.get_y_max() - self.get_y_min()


class RadarFeature(Feature):
    pass
