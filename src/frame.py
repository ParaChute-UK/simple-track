import datetime as dt
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndimage
from typing import Union
from feature import Feature
from utils import check_arrays, check_valid_ids


class FeaturesNotFoundError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class Frame:
    def __init__(self):
        self.time = None
        self.raw_field = None
        self.feature_field = None
        self.lifetime_field = None
        self.max_id = None
        self.features = {}
        self.y_flow = None
        self.x_flow = None

    def __repr__(self) -> str:
        repr_str = f"Event file_id: {self.file_id}, time: {self.time}, "
        repr_str += f"num_features: {len(self.features)}"
        return repr_str

    def __eq__(self, other) -> bool:
        if not isinstance(other, Frame):
            return False
        return self.time == other.time

    def import_time_and_data(self, time: dt.datetime, data: NDArray) -> None:
        self.raw_field = check_arrays(data, ndim=2)
        if not isinstance(time, dt.datetime):
            raise TypeError(
                f"Expected 'output_time' to be datetime objcet, got {type(time)}"
            )
        self.time = time

    def set_time(self, time: dt.datetime):
        self.time = time

    def get_time(self):
        return self.time

    def get_feature_field(self) -> NDArray[np.integer]:
        return self.feature_field

    def get_lifetime_field(self) -> NDArray:
        return self.lifetime_field

    def get_feature(self, feature_id: int) -> Feature:
        feature_id = check_valid_ids(feature_id)
        if feature_id in self.features.keys():
            return self.features[feature_id]
        else:
            return None

    def get_features(self) -> dict:
        return self.features

    def get_flow(self) -> Union[NDArray, None]:
        return self.y_flow, self.x_flow

    def set_feature_field(self, feature_field: NDArray) -> None:
        self.feature_field = feature_field

    def replace_features(self, new_features: dict) -> None:
        self.features = new_features

    def get_max_id(self) -> int:
        return self.max_id

    def set_max_id(self, max_id: int) -> None:
        max_id = check_valid_ids(max_id)
        self.max_id = max_id

    def identify_features(
        self, min_size: int, threshold: float, under_threshold: bool
    ) -> None:
        """
        Call the "label_features" function to identify distinct regions in the input field
        that meet a specified threshold condition.
        Then, analyses each of the identified features to find properties

        Args:
            - min_size (float): Minimum area (in number of grid points) for a region to be considered valid
            - threshold (float): Threshold value for identifying regions
            - under_threshold (bool): If True, regions under the threshold are considered;
                if False, regions over the threshold are considered.
        """
        if self.raw_field is None:
            raise Exception("Data has not been loaded into Frame")

        self.feature_field = label_features(
            field=self.raw_field,
            min_area=min_size,
            threshold=threshold,
            under_threshold=under_threshold,
        )
        # Provisionally set the lifetime field to 1 anywhere there is a feature
        self.lifetime_field = np.zeros_like(self.feature_field)
        self.lifetime_field[self.feature_field > 0] = 1
        self.max_id = int(np.max(self.feature_field))
        self.populate_features()

    def populate_features(self) -> None:
        # Check for existing features dict
        if self.features:
            self.features = {}

        if self.feature_field is None:
            return

        feature_ids = np.unique(self.feature_field)
        # Remove 0 from the list of ids (usually this is at idx 0 but can't guarantee this)
        feature_ids = np.delete(feature_ids, np.where(feature_ids == 0)[0][0])
        feature_ids = check_valid_ids(feature_ids)

        # Don't include 0 in Feature population, this is reserved for background
        for feature_id in feature_ids:
            # Get the pixel locations of the feature in the field
            # For 2D data, np.where returns two arrays containing y, x locations
            feature_mask = np.where(self.feature_field == feature_id)
            feature_coords = np.array(feature_mask)

            # Construct Feature object, set relevant properties, add to the list of features
            feature = Feature(
                id=feature_id, feature_coords=feature_coords, time=self.time
            )
            # If raw field is not None, use this to find max value within Feature
            if self.raw_field is not None:
                feature.extreme = max(self.raw_field[feature_mask])
            self.features[feature_id] = feature

    def assign_displacements(self, y_flow: NDArray, x_flow: NDArray) -> None:
        """
        Add flow field to frame. Use input y_flow and x_flow fields to assign
        dy and dx displacements to each Feature in the Frame

        Args:
            y_flow (NDArray): _description_
            x_flow (NDArray): _description_
        """
        if self.feature_field is None or not self.features:
            raise FeaturesNotFoundError(
                "Features have not been loaded into this Frame. Cannot assign displacements"
            )

        self.y_flow, self.x_flow = check_arrays(
            y_flow, x_flow, ndim=2, equal_shape=True
        )

        # Assign these displacements to each Feature in the Frame using
        # mean of flow field for each grid point spanning the Feature
        for feature_id, feature in self.features.items():
            feature_mask = self.feature_field == feature_id
            feature_dy = np.mean(y_flow[feature_mask])
            feature_dx = np.mean(x_flow[feature_mask])
            feature.dydx = (feature_dy, feature_dx)

    def get_next_available_feature_id(self) -> int:
        """
        Get the next available feature ID for this Frame.
        Used when new features are created.

        Returns:
            int: new id
        """
        if self.max_id is None:
            if self.feature_field is not None:
                self.max_id = np.max(self.feature_field).item()
            else:
                self.max_id = 0
        self.max_id += 1
        return self.max_id

    def promote_provisional_ids(self) -> None:
        """
        Promote provisional ids to final ids for all features.
        """
        # Construct updated features dictionary with new ids as keys
        new_features_dict = {}

        for feature in self.features.values():
            if feature.provisional_id is not None:
                feature.id = feature.provisional_id
                feature.provisional_id = None
            new_features_dict[feature.id] = feature

        self.features = new_features_dict

    def update_fields_using_provisional_ids(self) -> None:
        """
        Update the feature_field to reflect provisional ids.
        """
        if self.feature_field is None:
            raise FeaturesNotFoundError(
                "Feature field is not set. Cannot update using provisional ids."
            )

        if not self.features:
            raise FeaturesNotFoundError(
                "Features have not been loaded into this Frame. Cannot update using provisional ids."
            )

        updated_feature_field = np.zeros_like(self.feature_field)
        updated_lifetime_field = np.zeros_like(self.feature_field)

        for feature in self.features.values():
            feature_mask = self.feature_field == feature.id
            updated_lifetime_field[feature_mask] = feature.lifetime
            if feature.provisional_id is not None:
                updated_feature_field[feature_mask] = feature.provisional_id
            else:
                updated_feature_field[feature_mask] = feature.id

        self.feature_field = updated_feature_field
        self.lifetime_field = updated_lifetime_field

    def get_new_features(self) -> list:
        if not self.features:
            return []
        return [feature for feature in self.features.values() if feature.is_new()]

    def get_dissipating_features(self) -> list:
        if not self.features:
            return []
        return [
            feature for feature in self.features.values() if feature.is_dissipating()
        ]

    def get_init_field(self, centroid_only: bool = True) -> NDArray:
        return self.get_field("init", centroid_only)

    def get_dissipation_field(self, centroid_only: bool = True) -> NDArray:
        return self.get_field("dissipation", centroid_only)

    def get_field(self, field_type: str, centroid_only: bool = True) -> NDArray:
        feature_methods = {
            "init": self.get_new_features,
            "dissipation": self.get_dissipating_features,
        }
        if field_type not in feature_methods.keys():
            raise KeyError(f"field_type must be one of {feature_methods.keys()}")

        field = np.zeros_like(self.feature_field)
        for feature in feature_methods[field_type]():
            if centroid_only:
                # tuple to ensure correct indexing
                # Round centroid to nearest integer and cast to int
                centroid_coord = tuple(np.rint(feature.centroid).astype(int))
                field[centroid_coord] = 1
            else:
                # Populate field with full size of feature
                init_mask = self.feature_field == feature.id
                field[init_mask] = 1
        return field


class Timeline:
    def __init__(self):
        self.timeline = {}

    def add_to_timelime(self, frame: Frame):
        if not isinstance(frame, Frame):
            raise TypeError(f"Expected type Frame, got {type(frame)}")
        if frame.get_time() is None:
            raise ValueError("Frame time is not set. Cannot add to timeline.")
        self.timeline[frame.get_time()] = frame

    def get_previous_frame(self, current_time: dt.time) -> Frame:
        if len(self.timeline) == 0:
            raise ValueError("Timeline is empty. No previous frame to return.")
        if len(self.timeline) == 1:
            return None

        prev_times = [time for time in self.timeline.keys() if time < current_time]
        closest_time = max(prev_times) if prev_times else None
        if closest_time is None:
            raise ValueError("No previous frame found in timeline")
        return self.timeline[closest_time]

    def purge_old_frame(self, max_frames: int = 2) -> None:
        # Remove any frames that aren't needed anymore, as defined by max_frames
        pass

    def get_timeline(self):
        return self.timeline

    def get_frame(self, time: dt.datetime) -> Frame:
        if time not in self.timeline.keys():
            raise ValueError(f"No frame found for time {time}")
        return self.timeline[time]


def label_features(
    field: NDArray[np.floating],
    min_area: float,
    threshold: float,
    under_threshold: bool = False,
    connectivity_structure: NDArray[np.bool] = np.ones((3, 3)),
) -> NDArray[np.integer]:
    """
    Label distinct regions in the input field that meet a specified threshold condition.

    Args:
        field (np.ndarray):
            2D input array of data to be labelled
        min_area (float):
            Minimum area (in number of grid points) for a region to be considered valid
        threshold (float):
            Threshold value for identifying regions
        under_threshold (bool, optional):
            If True, regions under the threshold are considered;
            if False, regions over the threshold are considered.
            Defaults to False.
        connectivity_structure (NDArray, optional):
            Boolean array defining connectivity for region labelling.
            Default is 8-way connectivity, meaning all cardinal AND diagonal neighbours that
            meet the threshold condition are considered part of the same region.
            An alternative arrangement would be 4-way connectivity (diagonals omitted), defined as:
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
            See scipy.ndimage.label documentation for more details.
            Defaults to np.ones((3, 3)).

    Raises:
        TypeError: field must be a numpy ndarray
        ValueError: min_area must be a non-negative number
        ValueError: threshold must be a number"
        TypeError: under_threshold must be a boolean
        ValueError: field must be a 2D array

    Returns:
        NDArray[np.int_]: 2D Integer field of labelled regions, same shape as input field
    """

    # Check input types
    field = check_arrays(field, ndim=2)

    if not isinstance(min_area, (int, float)) or min_area < 0:
        raise ValueError("min_area must be a non-negative number")
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if not isinstance(under_threshold, bool):
        raise TypeError("under_threshold must be a boolean")

    # Construct feature field using threshold and threshold condition
    # Grid points meeting the condition are set to 1, others to 0
    if under_threshold:
        feature_field = np.where(field < threshold, 1, 0)
    else:
        feature_field = np.where(field > threshold, 1, 0)

    # Identify and label distinct regions in the feature field
    id_regions, num_ids = ndimage.label(feature_field, structure=connectivity_structure)

    # Any regions smaller than the min_area are removed from the feature field
    # before re-running feature labelling
    id_sizes = np.array(ndimage.sum(feature_field, id_regions, range(num_ids + 1)))
    area_mask = id_sizes < min_area
    feature_field[area_mask[id_regions]] = 0
    id_regions, num_ids = ndimage.label(feature_field, structure=connectivity_structure)

    return id_regions
