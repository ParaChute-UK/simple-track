import datetime as dt
from netCDF4 import Dataset as ncfile
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndimage
from typing import Union
from Feature import Feature


class Frame:
    def __init__(self):
        self.file_id = None
        # TODO: make time an input in init, remove set_time as an option
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

    def set_time(self, time: dt.datetime):
        self.time = time

    def get_time(self):
        return self.time

    def get_feature_field(self) -> NDArray[np.integer]:
        return self.feature_field

    def get_feature(self, feature_id: int) -> Feature:
        return self.features[feature_id]

    def get_features(self) -> dict:
        return self.features

    def get_flow(self) -> Union[NDArray, None]:
        return self.y_flow, self.x_flow

    def set_feature_field(self, feature_field: NDArray) -> None:
        self.feature_field = feature_field

    def replace_features(self, new_features: dict) -> None:
        self.features = new_features

    # TODO: add functionality for user-definable loading functions
    def load_data(self, filename: str) -> None:
        """
        Load data and extract time information

        Args:
            filename (str):
                Path to file
        """
        nc = ncfile(filename)
        data = nc.variables["var"][200:600, 250:550] / 32
        data = np.flipud(np.transpose(data))
        self.raw_field = data

        file_id = str(filename)[-9:-5]
        self.file_id = file_id
        self.time = dt.time(hour=int(file_id[0:2]), minute=int(file_id[2:4]))

    def identify_features(
        self, min_size: int, threshold: float, under_threshold: bool
    ) -> None:
        """
        Call the "label_storms" function to identify distinct regions in the input field
        that meet a specified threshold condition.
        Then, analyses each of the identified features to find properties

        Args:
            - min_size (float): Minimum area (in number of grid points) for a region to be considered valid
            - threshold (float): Threshold value for identifying regions
            - under_threshold (bool): If True, regions under the threshold are considered;
                if False, regions over the threshold are considered.
        """
        if self.raw_field is None:
            raise Exception("Data has not been loaded into Event.")

        self.feature_field = label_features(
            field=self.raw_field,
            min_area=min_size,
            threshold=threshold,
            under_threshold=under_threshold,
        )
        self.max_id = int(np.max(self.feature_field))
        self.populate_features()

    def populate_features(self) -> None:
        # Check for existing features dict
        if self.features:
            self.features = {}

        max_feature_id = int(np.max(self.feature_field))
        for feature_id in range(max_feature_id):
            # Get the pixel locations of the feature in the field
            # For 2D data, np.where returns two arrays containing y, x locations
            feature_coords = np.array(np.where(self.feature_field == feature_id))

            # Add this to the list of features
            self.features[feature_id] = Feature(
                id=feature_id, feature_coords=feature_coords, time=self.time
            )

    def assign_displacements(self, y_flow: NDArray, x_flow: NDArray) -> None:
        """
        Add flow field to frame. Use input y_flow and x_flow fields to assign
        dy and dx displacements to each Feature in the Frame

        Args:
            y_flow (NDArray): _description_
            x_flow (NDArray): _description_
        """
        if self.feature_field is None or not self.features:
            raise Exception(
                "Features have not been loaded into this Frame. Cannot assign displacements"
            )

        if not all(isinstance(flow, NDArray) for flow in [y_flow, x_flow]):
            raise ValueError("Inputs must be NDarrays")

        if not all(isinstance(flow.ndim == 2) for flow in [y_flow, x_flow]):
            raise ValueError("Inputs must be 2D arrays")

        if not all(flow.shape == self.feature_field.shape for flow in [y_flow, x_flow]):
            raise ValueError("Input flow must have same shape as self.feature_field")

        self.y_flow = y_flow
        self.x_flow = x_flow

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
        Used when new features are created via splitting.

        Returns:
            int: new id
        """
        if self.max_id is None:
            self.max_id = 0
        self.max_id += 1
        return self.max_id

    def promote_provisional_ids(self) -> None:
        """
        Promote provisional ids to final ids for all features in this Frame.
        """
        # Construct updated features dictionary with new ids as keys
        new_features_dict = {}

        for feature in self.features.values():
            if feature.provisional_id is not None:
                feature.id = feature.provisional_id
                feature.provisional_id = None
            else:
                feature.id = feature.id
            new_features_dict[feature.id] = feature

        self.features = new_features_dict

    def update_fields_using_provisional_ids(self) -> None:
        """
        Update the feature_field to reflect provisional ids.
        """
        if self.feature_field is None:
            raise Exception(
                "Feature field is not set. Cannot update using provisional ids."
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


class Timeline:
    def __init__(self):
        self.timeline = {}

    def add_to_timelime(self, frame: Frame):
        if not isinstance(frame, Frame):
            raise TypeError(f"Expected type Frame, got {type(frame)}")
        self.timeline[frame.get_time()] = frame

    def get_previous_frame(self, current_time: dt.time) -> Frame:
        # Return the event prior to the event at current_time
        pass

    def purge_old_frame(self, max_frames: int = 2) -> None:
        # Remove any frames that aren't needed anymore, as defined by max_frames
        pass


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
    if not isinstance(field, np.ndarray):
        raise TypeError("field must be a numpy ndarray")
    if not isinstance(min_area, (int, float)) or min_area < 0:
        raise ValueError("min_area must be a non-negative number")
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if not isinstance(under_threshold, bool):
        raise TypeError("under_threshold must be a boolean")

    # Check the input field is 2D
    if field.ndim != 2:
        raise ValueError("field must be a 2D array")

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
