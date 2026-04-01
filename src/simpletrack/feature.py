import datetime as dt
from typing import Union

import numpy as np
from numpy.typing import NDArray

from simpletrack.utils import check_arrays, check_valid_ids, native


class Feature:
    def __init__(
        self, id: int, feature_coords: NDArray[np.integer], time: dt.datetime
    ) -> None:
        check_arrays(feature_coords, ndim=2, dtype=int)
        id = check_valid_ids(id)
        self._id = native(id)
        self._provisional_id = None
        self._feature_coords = feature_coords
        self._time = time
        self._centroid = None
        self._lifetime = 1
        self._final_timestep = False
        self._accreted = None
        self._accreted_in_next_frame_by = None
        self._parent = None
        self._children = None
        self._dydx = ()
        self._extreme = None

    def __repr__(self) -> str:
        repr_str = f"Feature id: {self._id} (provisionally {self._provisional_id}), "
        repr_str += f"lifetime: {self._lifetime} timestep(s) at time: {self._time}"
        return repr_str

    def __eq__(self, other):
        if (
            self._time == other._time
            and self._id == other._id
            and np.array_equal(self._feature_coords, other._feature_coords)
        ):
            return True
        return False

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
    def accreted_in_next_frame_by(self) -> int:
        return self._accreted_in_next_frame_by

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
        return native(self._dydx)

    @property
    def extreme(self) -> float:
        return self._extreme

    @coords.setter
    def coords(self, new_coords: NDArray[np.integer]) -> None:
        self._feature_coords = new_coords
        self._centroid = self.calculate_centroid()  # Update centroid when coords change

    @parent.setter
    def parent(self, parent_id: int) -> None:
        if parent_id is None:
            self._parent = None
            return

        parent_id = check_valid_ids(parent_id)
        self._parent = native(parent_id)

    @children.setter
    def children(self, child_ids: list[int]) -> None:
        if child_ids is None:
            self._children = None
            return

        child_ids = check_valid_ids(child_ids)
        if isinstance(child_ids, int):
            self._children = native([child_ids])
        elif isinstance(child_ids, (list, tuple, np.ndarray)):
            self._children = native(child_ids)
        else:
            raise TypeError(
                "children must be set to an int or list/tuple/NDArray of ints"
            )

    @dydx.setter
    def dydx(self, dy_dx: tuple) -> None:
        self._dydx = native(dy_dx)

    @id.setter
    def id(self, _id: int) -> None:
        _id = check_valid_ids(_id)
        self._id = native(_id)

    @lifetime.setter
    def lifetime(self, lifetime: int) -> None:
        self._lifetime = native(lifetime)

    @provisional_id.setter
    def provisional_id(self, _id: int) -> None:
        if _id is not None:
            _id = check_valid_ids(_id)
        self._provisional_id = native(_id)

    @accreted.setter
    def accreted(self, accreted_ids: Union[int, list, None]) -> None:
        if accreted_ids is None:
            self._accreted = None
            return

        accreted_ids = check_valid_ids(accreted_ids)
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

    @accreted_in_next_frame_by.setter
    def accreted_in_next_frame_by(self, id_of_accreting_feature: int):
        id_of_accreting_feature = check_valid_ids(id_of_accreting_feature)
        self._accreted_in_next_frame_by = id_of_accreting_feature

    @extreme.setter
    def extreme(self, extreme_val: float) -> None:
        self._extreme = extreme_val

    def calculate_centroid(self) -> tuple:
        y_centroid = native(np.mean(self._feature_coords[0, :]))
        x_centroid = native(np.mean(self._feature_coords[1, :]))
        return (y_centroid, x_centroid)

    def accrete_ids(self, feature_ids: int | list[int]) -> None:
        feature_ids = check_valid_ids(feature_ids)
        if self._accreted is None:
            self._accreted = []
        if isinstance(feature_ids, int):
            self._accreted.append(native(feature_ids))
        elif isinstance(feature_ids, np.ndarray):
            self._accreted.extend(feature_ids.tolist())
        else:
            self._accreted.extend(feature_ids)

    def spawns(self, feature_ids: int | list[int]) -> None:
        feature_ids = check_valid_ids(feature_ids)
        if self._child is None:
            self._child = []
        if isinstance(feature_ids, int):
            self._child.append(feature_ids)
        else:
            self._child.extend(feature_ids)

    def get_size(self) -> int:
        return len(self._feature_coords[0])

    def set_as_final_timestep(self) -> None:
        self._final_timestep = True

    def summarise(self, output_type="str", headers_only=False):
        summary = {
            "id": self._id,
            "centroid": self.centroid,
            "size": self.get_size(),
            # native() does not convert dydx to python type for some reason
            "dydx": tuple([val.item() for val in self._dydx]),
            "extreme": self._extreme,
            "lifetime": self._lifetime,
            "accreted": self._accreted,
            # This will not be output properly in the current workflow, since each output occurs
            # after the current frame analysis has finished, but this can only be set after comparison
            # with the next frame of data.
            # "accredted_in_next_frame_by": self._accreted_in_next_frame_by,
            "parent": self._parent,
            "children": self._children,
        }
        if headers_only:
            return list(summary.keys())
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

    def is_dissipating(self) -> bool:
        """
        Returns bool whether the Feature is 'dissipating' in the sense that
        this is its final timestep AND it has not been accreted by another Feature
        """
        if self._final_timestep and self._accreted_in_next_frame_by is None:
            return True
        return False

    def is_final_timstep(self) -> bool:
        """
        Returns bool whether this is the final timestep for the Feature, i.e., it is either
        dissipating or accreting into another Feature
        """
        return self._final_timestep
