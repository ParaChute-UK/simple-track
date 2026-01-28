from numpy.typing import NDArray
import numpy as np
from typing import Union
import datetime as dt
from utils import check_arrays, native


class Feature:
    def __init__(
        self, id: int, feature_coords: NDArray[np.integer], time: dt.datetime
    ) -> None:
        check_arrays(feature_coords, ndim=2, dtype=int)
        self._id = native(id)
        self._provisional_id = None
        self._feature_coords = feature_coords
        self._time = time
        self._centroid = None
        self._lifetime = 1
        self._accreted = None
        self._parent = None
        self._children = None
        self._dydx = ()

    def __repr__(self) -> str:
        repr_str = f"Feature id: {self.id} (provisionally {self.provisional_id}), "
        repr_str += f"lifetime: {self.lifetime} timestep(s) at time: {self.time}"
        return repr_str

    @property
    def id(self) -> int:
        return self._id

    @property
    def provisional_id(self) -> int:
        return self._provisional_id

    @property
    def centroid(self) -> tuple:
        if self._centroid is None:
            self._centroid = self.calculate_centroid()
        return self._centroid

    @property
    def time(self) -> dt.datetime:
        return self._time

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
    def children(self) -> list[int]:
        return self._children

    @property
    def dydx(self) -> tuple:
        """
        Feature displacement valid at the given Frame time

        Returns:
            tuple: (dy, dx)
        """
        return self._dydx

    @coords.setter
    def coords(self, new_coords: NDArray[np.integer]) -> None:
        self._feature_coords = new_coords
        self._centroid = self.calculate_centroid()  # Update centroid when coords change

    @parent.setter
    def parent(self, parent_id: int) -> None:
        self._parent = native(parent_id)

    @children.setter
    def children(self, child_ids: list[int]) -> None:
        if isinstance(child_ids, int):
            self._children = native([child_ids])
        elif isinstance(child_ids, list):
            self._children = native(child_ids)
        else:
            raise TypeError("children must be set to an int or list of ints")

    @dydx.setter
    def dydx(self, dy_dx: tuple) -> None:
        self._dydx = native(dy_dx)

    @id.setter
    def id(self, id: int) -> None:
        self._id = native(id)

    @lifetime.setter
    def lifetime(self, lifetime: int) -> None:
        self._lifetime = native(lifetime)

    @provisional_id.setter
    def provisional_id(self, id: int) -> None:
        self._provisional_id = native(id)

    @accreted.setter
    def accreted(self, accreted_ids: Union[int, list]) -> None:
        if isinstance(accreted_ids, int):
            self._accreted = [accreted_ids]
        elif isinstance(accreted_ids, np.ndarray):
            self._accreted = accreted_ids.tolist()
        elif isinstance(accreted_ids, tuple):
            self._accreted = list(accreted_ids)
        elif isinstance(accreted_ids, list):
            self._accreted = accreted_ids
        else:
            msg = f"Expected list, tuple, NDArray or int, got f{type(accreted_ids)}"
            raise TypeError(msg)

        if len(self._accreted) < 1:
            self._accreted = None

    def calculate_centroid(self) -> tuple:
        y_centroid = native(np.mean(self._feature_coords[0, :]))
        x_centroid = native(np.mean(self._feature_coords[1, :]))
        return (y_centroid, x_centroid)

    def accretes(self, feature_ids: int | list[int]) -> None:
        if self._accreted is None:
            self._accreted = []
        if isinstance(feature_ids, int):
            self._accreted.append(native(feature_ids))
        elif isinstance(feature_ids, np.ndarray):
            self._accreted.extend(feature_ids.tolist())
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

    def get_size(self) -> int:
        return len(self._feature_coords[0])

    def summarise(self, output_type="str"):
        summary = {
            "id": self.id,
            "centroid": self.centroid,
            "size": self.get_size(),
            "dydx": self.dydx,
            "lifetime": self.lifetime,
            "accreted": self.accreted,
            "parent": self.parent,
            "children": self.children,
        }
        if output_type == "str":
            return str(summary)
        elif output_type == "dict":
            return summary
        else:
            raise ValueError("output_type must be 'str' or 'dict'")

    def is_new(self) -> bool:
        """
        Returns bool whether the Feature is 'new' in the sense that it
        has lifetime of 1 AND it has not split from another Feature
        (i.e., it has no parent)
        """
        if self._lifetime == 1 and self._parent is None:
            return True
        return False
