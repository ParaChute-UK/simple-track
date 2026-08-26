import datetime as dt

import numpy as np
from numpy.typing import NDArray

from simpletrack.utils import check_arrays, check_valid_ids, native


class Feature:
    """
    Object containing details about a specific feature, including its id, time,
    centroid, maximum value, lifetime, and whether it has undergone any
    mergers of splits in the current timestep.
    """

    def __init__(
        self, id: int, feature_coords: NDArray[np.integer], time: dt.datetime
    ) -> None:
        self._feature_coords = check_arrays(feature_coords, dtype=int)
        if self._feature_coords.ndim > 2:
            msg = (
                "feature_coords must be a 2D array of shape (2, n)"
                + " or 1D array of shape (2)"
            )
            raise ValueError(msg)
        id = check_valid_ids(id)
        self._id = native(id)
        self._provisional_id = None
        self._time = time
        self._centroid = None
        self._lifetime = 1
        self._final_timestep = False
        self._accreted = []
        self._accreted_in_next_frame_by = None
        self._parent = None
        self._children = []
        self._dydx = ()
        self._max = None
        self._mean = None
        # Only attempt pca decomposition if feature is larger than 1 pixel
        if self._feature_coords.ndim > 1 and self._feature_coords.shape[-1] > 1:
            (
                self._major_vector,
                self._minor_vector,
                self._major_radius,
                self._minor_radius,
            ) = self.calculate_major_minor_axes(self._feature_coords)

    def __repr__(self) -> str:
        repr_str = f"Feature id: {self._id}, "
        repr_str += f"lifetime: {self._lifetime} timestep(s) at time: {self._time}"
        return repr_str

    def __eq__(self, other):
        return (
            self._time == other._time
            and self._id == other._id
            and np.array_equal(self._feature_coords, other._feature_coords)
        )

    @property
    def id(self) -> int:
        """
        The id of the Feature, a positive nonzero integer
        """
        return self._id

    @property
    def provisional_id(self) -> int:
        """
        A provisional id for the Feature, used when matching Features between Frames
        before final id assignement.
        """
        return self._provisional_id

    @property
    def centroid(self) -> tuple:
        """
        Central point of the Feature, calculated as the mean of all y, x
        coordinates spanned by the Feature
        """
        if self._centroid is None:
            self._centroid = self.calculate_centroid()
        return self._centroid

    @property
    def time(self) -> dt.datetime:
        """
        Time that the Feature exists at.
        """
        return self._time

    @property
    def coords(self) -> NDArray[np.integer]:
        """
        Coordinates spanned by the Feature, as a 2D array of shape (2, n),
        where the first row contains y coordinates, and the second
        row contains x coordinates.
        """
        return self._feature_coords

    @property
    def lifetime(self) -> int:
        """
        Lifetime that the current Feature has existed for.
        If lifetime = 1, it initiated at the current timestep.
        """
        return self._lifetime

    @property
    def accreted(self) -> list[int]:
        """
        List of Feature ids that have been accreted by this Feature in the
        current timestep, if any. Return None if no accreted features
        """
        if len(self._accreted) < 1:
            return None
        return self._accreted

    @property
    def accreted_in_next_frame_by(self) -> int:
        """
        ID of Feature that accretes this Feature in the next frame, if any.
        This will not be known until the next frame of data has been processed.
        """
        return self._accreted_in_next_frame_by

    @property
    def parent(self) -> int:
        """
        If this Feature split from another Feature in the current timestep,
        this is the ID of the parent Feature, Otherwise, this is None.
        """
        return self._parent

    @property
    def children(self) -> list[int]:
        """
        If other Features split from this Feature in the current timestep,
        this is the list of IDs of those child Features. Otherwise this is None
        """
        if len(self._children) < 1:
            return None
        return self._children

    @property
    def dydx(self) -> tuple:
        """
        Motion vector that translated the current Feature from its position in the
        previous frame to its position in the current frame. This is calculated from
        the mean of y_flow, x_flow values spanned by the Feature in the
        frame with the same timestamp.
        """
        return native(self._dydx)

    @property
    def max(self) -> float:
        """
        Maximum value of the Feature in the raw input data
        """
        return self._max

    @property
    def mean(self) -> float:
        """
        Mean value of the Feature in the raw input data
        """
        return self._mean

    @property
    def major_vector(self) -> NDArray:
        """
        Unit vector of semi-major axis in (y, x) format
        """
        return self._major_vector

    @property
    def minor_vector(self) -> NDArray:
        """
        Unit vector of semi-minor axis in (y, x) format
        """
        return self._minor_vector

    @property
    def major_radius(self) -> float:
        """
        Radius of the semi-major axis
        """
        return self._major_radius

    @property
    def minor_radius(self) -> float:
        """
        Radius of semi-minor axis
        """
        return self._minor_radius

    @coords.setter
    def coords(self, new_coords: NDArray[np.integer]) -> None:
        self._feature_coords = check_arrays(new_coords, dtype=int)
        self._centroid = self.calculate_centroid()  # Update centroid when coords change
        # Update feature elongation information
        # Only attempt pca decomposition if feature is larger than 1 pixel
        if new_coords.ndim == 2:
            (
                self._major_vector,
                self._minor_vector,
                self._major_radius,
                self._minor_radius,
            ) = self.calculate_major_minor_axes(self._feature_coords)
        elif new_coords.ndim == 1:
            self._major_vector = None
            self._minor_vector = None
            self._major_radius = None
            self._minor_radius = None
        else:
            raise ValueError(
                "feature_coords must be a 1D or 2D array of shape (2, n) or (2)"
            )

    @parent.setter
    def parent(self, parent_id: int) -> None:
        if parent_id is None:
            self._parent = None
            return

        parent_id = check_valid_ids(parent_id)
        self._parent = native(parent_id)

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

    @accreted_in_next_frame_by.setter
    def accreted_in_next_frame_by(self, id_of_accreting_feature: int):
        id_of_accreting_feature = check_valid_ids(id_of_accreting_feature)
        self._accreted_in_next_frame_by = id_of_accreting_feature

    @max.setter
    def max(self, max_val: float) -> None:
        self._max = max_val

    @mean.setter
    def mean(self, mean_val: float) -> None:
        self._mean = mean_val

    def calculate_major_minor_axes(
        self, coords: NDArray
    ) -> list[NDArray, NDArray, float, float]:
        """
        Calculate major and minor axes of the feature using principal component
        analysis. Returns

        Args:
            coords (np.NDArray): 2D array of coordinates of shape (2, N) spanned by
            the feature, with format (y, x), i.e., first row contains y coordinates,
            and second row contains x coordinates

        Returns:
            list[np.NDArray, np.NDArray, float, float]:
                Semi-major axis vector:
                    NDArray of shape (2) containing (y, x) unit vector
                    (Eigenvector of largest eigenvalue)
                Semi-minor axis vector:
                    NDArray of shape (2) containing (y, x) unit vector
                    (Eigenvector of smallest eigenvalue)
                Semi-major axis radius:
                    float: Radius of the semi-major axis
                Semi-minor axis radius:
                    float: Radius of semi-minor axis
        """
        coords = check_arrays(coords, ndim=2, dtype=int)

        # First, subtract mean from coords
        mean_coords = np.mean(coords, axis=1)
        # Need to add new axis to make shape of mean_coords the same as coords
        mean_adjusted_coords = coords - mean_coords[:, np.newaxis]

        # Calculate covariance matrix, eigenvalues, eigenvectors
        cov_mat = np.cov(mean_adjusted_coords)
        evals, evecs = np.linalg.eig(cov_mat)

        # Use evals to determine sort order of eiegenvectors
        sort_idx = np.argsort(evals)[::-1]

        # In this case, it is difficult to relate eigenvalues to major/minor
        # radii, since this seems to depend on the feature shape.
        # Instead, project coords onto eigenvectors and find difference
        # between max and min to get diameter of feature.
        # Method follows https://math.stackexchange.com/questions/4231542/

        major_unit_vector = evecs[:, sort_idx[0]]
        minor_unit_vector = evecs[:, sort_idx[1]]

        major_eig_proj = major_unit_vector.T @ coords
        minor_eig_proj = minor_unit_vector.T @ coords

        major_diameter = np.max(major_eig_proj) - np.min(major_eig_proj)
        minor_diameter = np.max(minor_eig_proj) - np.min(minor_eig_proj)

        # Reverse eigenvectors to be in (y,x) format, consistent with rest of package
        # Native converts from numpy type to python type
        # Real ensures types are not complex, which can happen with PCA decomposition
        return (
            native(np.real(major_unit_vector[::-1])),
            native(np.real(minor_unit_vector[::-1])),
            native(np.real(major_diameter / 2)),
            native(np.real(minor_diameter / 2)),
        )

    def calculate_centroid(self) -> tuple:
        """
        Calculate centroid of the Feature as the mean of all y, x coordinates
        spanned by the Feature

        Returns:
            tuple: (y_centroid, x_centroid)
        """
        if self._feature_coords.ndim == 1:
            return (self._feature_coords[0], self._feature_coords[1])
        elif self._feature_coords.ndim == 2:
            y_centroid = native(np.mean(self._feature_coords[0, :]))
            x_centroid = native(np.mean(self._feature_coords[1, :]))
            return (y_centroid, x_centroid)
        else:
            raise ValueError(
                "feature_coords must be a 1D or 2D array of shape (2, n) or (2)"
            )

    def accrete_ids(self, feature_ids: int | list[int], replace: bool = False) -> None:
        """
        Add input ids to the list of accreted_ids contained in this Feature.

        Args:
            feature_ids (int | list[int]):
                ID or list of IDs of features to be added to the accreted list for
                this Feature
            replace (bool, optional):
                If True, replaces existing accreted ids with the input ids,
                rather than adding inputs to the existing list.
                Defaults to False.
        """
        feature_ids = check_valid_ids(feature_ids)
        existing_ids = [] if replace else self._accreted

        if isinstance(feature_ids, int):
            existing_ids.append(native(feature_ids))
        elif isinstance(feature_ids, np.ndarray):
            existing_ids.extend(feature_ids.tolist())
        else:
            existing_ids.extend(feature_ids)

        self._accreted = existing_ids

    def spawns(self, feature_ids: int | list[int], replace: bool = False) -> None:
        """
        Add input ids to the list of child ids for this Feature.
        If replace is True, replaces existing child ids with the input ids,
        rather than adding to existing list.
        """
        feature_ids = check_valid_ids(feature_ids)
        existing_ids = [] if replace else self._children

        if isinstance(feature_ids, int):
            existing_ids.append(feature_ids)
        elif isinstance(feature_ids, np.ndarray):
            existing_ids.extend(feature_ids.tolist())
        else:
            existing_ids.extend(feature_ids)

        self._children = existing_ids

    def get_size(self) -> int:
        """
        Get number of pixels spanned by the Feature, calculated as the number of
        coordinate pairs in feature_coords array
        """
        return self._feature_coords.shape[1] if self._feature_coords.ndim == 2 else 1

    def set_as_final_timestep(self) -> None:
        """
        Set the feature as being in its final timestep, which means it is either
        accreting into another feature in the next frame, or it is dissipating
        in the next frame.
        """
        self._final_timestep = True

    def summarise(
        self, output_type: str = "str", headers_only: bool = False
    ) -> str | dict | list:
        """
        Return a summary of the Feature properties

        Args:
            output_type (str, optional):
                Format to return the summary in. Either "str" or "dict"
                Defaults to "str".
            headers_only (bool, optional):
                Whether to return just the headers of the summary
                (i.e., the keys of the summary dict)
                Defaults to False.
        """
        summary = {
            "id": self._id,
            "centroid": self.centroid,
            "size": self.get_size(),
            # native() does not convert dydx to python type for some reason
            "dydx": self.dydx,
            "max": self._max,
            "mean": self._mean,
            "lifetime": self._lifetime,
            "accreted": self._accreted,
            # This will not be output properly in the current workflow, since each
            # output occurs after the current frame analysis has finished, but this
            # can only be set after comparison with the next frame of data.
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
        return self._lifetime == 1 and self._parent is None

    def is_dissipating(self) -> bool:
        """
        Returns bool whether the Feature is 'dissipating' in the sense that
        this is its final timestep AND it has not been accreted by another Feature
        """
        return self._final_timestep and self._accreted_in_next_frame_by is None

    def is_final_timestep(self) -> bool:
        """
        Returns bool whether this is the final timestep for the Feature, i.e., it is
        either dissipating or accreting into another Feature
        """
        return self._final_timestep
