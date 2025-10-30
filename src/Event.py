import datetime
from netCDF4 import Dataset as ncfile
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndimage

from Feature import Feature


class Event:
    def __init__(self):
        self.file_id = None
        self.time = None

        self.raw_field = None
        self.feature_field = None
        self.features = {}

    def get_time(self):
        return self.time

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
        self.time = datetime.time(hour=int(file_id[0:2]), minute=int(file_id[2:4]))

    def identify_features(self, feature_config: dict) -> None:
        """
        Call the "label_storms" function to identify distinct regions in the input field
        that meet a specified threshold condition.
        Then, analyses each of the identified features to find properties

        Args:
            feature_config (dict):
                Dict of properties determining definition of features that includes:
                - min_size (float): Minimum area (in number of grid points) for a region to be considered valid
                - threshold (float): Threshold value for identifying regions
                - under_threshold (bool): If True, regions under the threshold are considered;
                  if False, regions over the threshold are considered.
        """
        if self.raw_field is None:
            raise Exception("Data has not been loaded into Event.")

        self.feature_field = label_features(
            field=self.raw_field,
            min_area=float(feature_config["min_size"]),
            threshold=float(feature_config["threshold"]),
            under_threshold=bool(feature_config["under_threshold"]),
        )

        max_feature_id = int(np.max(self.feature_field))
        for feature_id in range(max_feature_id):
            # Get the pixel locations of the feature in the field
            # For 2D data, np.where returns two arrays containing y, x locations
            feature_coords = np.array(np.where(self.feature_field == feature_id))

            # Add this to the list of features
            self.features[feature_id] = Feature(id=feature_id, feature_coords=feature_coords)


class RadarEvent(Event):
    def __init__(self):
        super().__init__()


class EventTimeline:
    def __init__(self):
        self.timeline = {}

    def add_to_timelime(self, event: Event):
        if not isinstance(event, Event):
            raise TypeError(f"Expected type Event, got {type(event)}")
        self.timeline[event.get_time()] = event

    def purge_old_events(self, )


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
    print("num_ids = ", num_ids)

    return id_regions
