import numpy as np
from numpy.typing import NDArray
from typing import Union
from collections import defaultdict
from Frame import Frame
from Feature import Feature


class FrameTracker:
    def __init__(
        self,
        overlap_nbhood: int = 5,
        overlap_threshold: float = 0.6,
    ):
        self.overlap_nbhood = overlap_nbhood
        self.overlap_threshold = overlap_threshold

    def run(self, prev_frame: Frame, current_frame: Frame) -> None:
        if not all(isinstance(frame, Frame) for frame in [prev_frame, current_frame]):
            raise TypeError(
                f"Expected type Frame, got {type(prev_frame)} and {type(current_frame)}"
            )

        # Step 1: Advect features in the previous frame using its flow field
        advected_frame = self.advect_frame(prev_frame)

        # Step 2: Features in the current Frame will have an id that is not related to the
        # previous/advected Frame in anyway.
        # Match features between the advected frame and the current frame by assigning a
        # new, proviosonal id to each Feature in the current Frame based on overlap
        self.match_advected_and_current_frame_features(advected_frame, current_frame)

        # Step 3: After Feature matching, there may be multiple Features in current Frame that were matched
        # to the same previous feature.
        # These features will now have the same ids. Find these ids and distinguish the most appropriate
        # match (to retain the id) from the other matches (which will be designated as children and
        # given a new id.)
        self.identify_parent_and_child_features(advected_frame, current_frame)

        # Step 4: Now that there is self consistent data in current frame, use this to
        # produce updated fields
        self.produce_fields(current_frame)

    def identify_parent_and_child_features(
        self, advected_frame: Frame, current_frame: Frame
    ) -> None:
        # TODO: rewrite the below! it is super convoluted
        # Instead, do a hist of matched_ids using all_matched_ids.values()
        # Then, do a {matched_id: no_of_matches} and retain all with no_of_matches > 1
        # Then loop over

        # The temporary id keys in current_frame.get_features() are likely mismatched with the id
        # of their Feature, but do serve as unique identifiers for them.
        # So, can still match Features with the same id
        # E.g. {0:8, 1:4, 2:8, 3:6} where temp_id:matched_id
        all_matched_ids = {
            temp_id: feature.id
            for temp_id, feature in current_frame.get_features().items()
        }

        # Flip this dictionary so that each matched_id is now a key, and each value is now a list of
        # dict keys (temp_ids) for finding that feature in the get_features() dict
        # E.g {4:[1], 6:[3], 8:[0,2]}
        matched_id_key_temp_id_val = defaultdict(list)
        for val, key in all_matched_ids:
            matched_id_key_temp_id_val[key].append(val)

        # Filter dict to retain only matched ids with multiple temp ids
        # E.g. {8:[0,2]}
        matched_ids_with_multiple_temp_ids = {
            key: vals
            for key, vals in matched_id_key_temp_id_val.items()
            if len(vals) > 1
        }

        # Loop over all matched ids to determine which Feature should retain this matched id
        # and which should be designated as children
        for matched_id, temp_id_list in matched_ids_with_multiple_temp_ids.items():
            pass
            # Check the overlap between the matched id in the advected frame and each
            # feature that matched to it in the current frame using the temp_id_list

    def produce_fields(self, frame: Frame) -> None:
        feature_field = frame.get_feature_field()
        updated_feature_field = np.zeros_like(feature_field)
        updated_feature_dict = {}

        # Initialise lifetime containing array
        lifetime_field = np.zeros_like(feature_field)

        # # Construct fields
        # # Constuct mask for assigning feature properties to fields
        # feature_mask = current_feature_field == feature_id
        # lifetime_field[feature_mask] = current_feature.lifetime
        # updated_feature_field[feature_mask] = current_feature.id

    def match_advected_and_current_frame_features(
        self, advected_frame: Frame, current_frame: Frame
    ) -> None:
        # Get the feature fields to analyse
        advected_feature_field = advected_frame.get_feature_field()
        current_feature_field = current_frame.get_feature_field()

        # Initiliase the id to set for a new feature
        new_feature_id = np.max(current_feature_field) + 1

        # Attempt to match features in the advected frame with current frame
        for feature_id, current_feature in current_frame.get_features().items():
            if not isinstance(current_feature, Feature):
                raise TypeError(f"Expected Feature, got {type(current_feature)}")

            # Count the ids contained in advected_feature_field that are in the same
            # position as the feature_id in the current_feature_field (normalised by
            # the feature sizes)
            overlap_hist = self.calculate_overlap_histogram(
                advected_feature_field, current_feature_field, feature_id, nbhood=None
            )

            # If the maximum overlap is not achieved, rerun with a nbhood surrouding the
            # feature centroid.
            if np.max(overlap_hist) < self.overlap_threshold:
                overlap_hist = self.calculate_overlap_histogram(
                    advected_feature_field,
                    current_feature_field,
                    feature_id,
                    nbhood=self.overlap_nbhood,
                )

            # Get the closest matching feature id to advected field, and any other ids
            # that have a sufficient overlap
            matching_id, other_sufficient_ids = self.find_ids_of_closest_overlaps(
                overlap_hist, advected_feature_field, current_feature_field, feature_id
            )

            # If a matching feature couldn't be found, this is a new Feature
            if matching_id is None:
                matching_id = new_feature_id
                new_feature_id += 1
                current_feature.lifetime = 1
            else:
                # Inherit lifetime from matching feature
                matching_feature = advected_frame.get_feature(matching_id)
                current_feature.lifetime = matching_feature.lifetime + 1
                # TODO: any other values to inherit here?

            # Assign the matching_id to this feature
            current_feature.id = matching_id

            # Determine whether this Feature has accreted other Features from the previous Frame
            if other_sufficient_ids is not None:
                current_feature.accretes(other_sufficient_ids)
                # TODO: understand what's going on with the sanity check part of the previous code.
                # lines 946-966

    def find_ids_of_closest_overlaps(
        self,
        overlap_hist: NDArray,
        advected_feature_field: NDArray,
        current_feature_field: NDArray,
        current_feature_id: int,
    ) -> list[Union[int, None], Union[NDArray, None]]:
        """
        Use overlap histogram to find the closest matching feature id in the advected field
        for the current_feature_id in the current_field. Any other ids that are also a sufficient
        overlap with the current feature are also separately returned.

        - If there are no sufficient overlaps, return None, None

        - If there is one sufficient overlap, return the label of the matching Feature
        from the advected field. Other sufficient ids is None

        - If there is more than one sufficient overlap, find the Feature label with the maximum overlap
        If there is more than one Feature with a maximum overlap, find the Feature from these that is
        closest to the centroid of the current Feature. If there is still more than one suitable
        Feature, choose the one with the smallest label. All sufficient ids that are not chosen as
        the matching id are also returned in an NDArray

        Args:
            overlap_hist (NDArray):
                Histogram of overlaps produced using calculate_overlap_histogram
            advected_feature_field (NDArray):
                Feature field from previous timestep advected by flow
            current_feature_field (NDArray):
                Feature field from current timestep
            current_feature_id (int):
                Feature ID in the current field to match with the previous field

        Returns:
            Union[int, None]:
                The new label to assign to the Feature. If None, there is no overlap
            Union[NDArray, None]:
                Any other labels that were a sufficient match but were not chosen as the
                best overlap. If None, there are no other sufficient ids.

        """
        # Check number of sufficient overlaps.
        sufficient_overlaps = overlap_hist >= self.overlap_threshold
        len_sufficient_overlaps = np.count_nonzero(sufficient_overlaps)

        # Setup returning variable that will only be not None if there are multiple overlaps
        other_sufficient_ids = None

        if len_sufficient_overlaps == 0:
            matching_id = None

        if len_sufficient_overlaps == 1:
            matching_id = np.argmax(overlap_hist)

        # If there is more than one sufficient overlap, keep the properties of the feature
        # with the largest overlap. If multiple have overlaps, keep nearest in centroid
        if len_sufficient_overlaps > 1:
            # Check for number of ids that share a maximum overlap
            max_overlaps = np.argwhere(overlap_hist == np.max(overlap_hist)).squeeze()

            if max_overlaps.size == 1:
                matching_id = np.argmax(overlap_hist)
            else:
                # Get the closest centroid for each feature sharing a maximum overlap
                centroid_distances = []
                current_feature_centroid = get_centroid(
                    current_feature_field, current_feature_id
                )
                for overlap_id in max_overlaps:
                    overlap_id_centroid = get_centroid(
                        advected_feature_field, overlap_id
                    )
                    distance = np.linalg.norm(
                        current_feature_centroid - overlap_id_centroid
                    )
                    centroid_distances.append(distance)

                # Find the feature that has the minimum distance. If there are still more
                # than 1 possible options at this stage, argmin returns the first instance
                min_distance_idx = np.argmin(centroid_distances)
                matching_id = max_overlaps[min_distance_idx]

            # Add the other sufficient overlaps to other_sufficient_ids
            # To ensure the matching id is now not included in other sufficient ids,
            # set sufficient overlaps to False at this id
            sufficient_overlaps[matching_id] = False
            other_sufficient_ids = np.argwhere(sufficient_overlaps).squeeze()

        return matching_id, other_sufficient_ids

    def calculate_overlap_histogram(
        self,
        advected_feature_field: NDArray,
        current_feature_field: NDArray,
        feature_id: int,
        nbhood: int = 0,
    ) -> NDArray:
        """
        Calculate the amount of overlap between two feature fields at the requested feature id.
        Method creates a mask containing areas of the current feature field containing the
        requested feature id. This mask can optionally be expanded using a nbhood surrouding the
        centroid of this feature. This mask is then used to select the same locations of the
        advected feature field. A histogram is produced giving the number of feature_ids contained
        in this reigon in the adveced feature field. This is normalised by the pixel size of the
        feature in each input field.

        Args:
            advected_feature_field (NDArray): _description_
            current_feature_field (NDArray): _description_
            feature_id (int): _description_
            nbhood (bool, optional): _description_. Defaults to False.

        Returns:
            NDArray: _description_
        """
        # Create feature mask using current feature field, or expand mask using a nbhood
        # if this is flagged in input
        feature_mask = np.where(current_feature_field == feature_id, True, False)
        if nbhood:
            centroid = get_centroid(current_feature_field, feature_id)
            feature_mask += generate_radial_mask(
                current_feature_field, centroid, nbhood
            )

        # Setup bins for comparing feature fields using histogram
        # Need to find max value among both input fields
        input_fields = [advected_feature_field, current_feature_field]
        max_val = np.max(input_fields)
        bins = np.arange(int(max_val) + 2)

        # Find overlap between two feature fields by finding histogram of points
        # using mask of current features applied to the advected feature field
        overlap_hist = np.histogram(advected_feature_field[feature_mask], bins)[0]

        # Set the first value of the hist to 0, since this represents the background
        overlap_hist[0] = 0

        # Normalise by the size of Feature within the mask of each field, if it exists
        sizes = [np.count_nonzero(field == feature_id) for field in input_fields]
        overlap_normed = [overlap_hist / fsize for fsize in sizes if fsize != 0]
        return np.mean(overlap_normed, axis=0)

    def advect_frame(self, frame: Frame) -> Frame:
        """
        Construct a new Frame with all Features in the input Frame advected by the given flow field

        Args:
            frame (Frame): Frame containing Features and a flow field

        Returns:
            Frame: advected Frame
        """

        if not isinstance(frame, Frame):
            raise TypeError(f"Expected 'Frame', got {type(frame)}")

        y_flow, x_flow = frame.get_flow()
        if y_flow is None or x_flow is None:
            print(f"y_flow: {y_flow}")
            print(f"x_flow: {x_flow}")
            raise ValueError(
                "y_flow and/or x_flow not defined. Cannot calculate advected Frame"
            )

        feature_field = frame.get_feature_field()
        advected_feature_field = advect_field_using_motion_vectors(
            feature_field, y_flow, x_flow
        )

        advected_frame = Frame()
        advected_frame.set_feature_field(advected_feature_field)
        advected_frame.populate_features()
        return advected_frame


def advect_field_using_motion_vectors(
    field: NDArray, y_flow: NDArray, x_flow: NDArray
) -> NDArray:
    """
    A (perhaps temporary) function that takes features (non-zero elements) in the 2D input field
    and advects them using the given motion vectors. This function performs this advection feature
    by feature, i.e., moves all contiguous elements of data by the same amount, retaining their shape.
    Code handles conflicts from multiple non-zero elements being advected to the same position by
    choosing the closest label centroid.

    Args:
        field (NDArray): _description_
        y_flow (NDArray): _description_
        x_flow (NDArray): _description_

    Returns:
        NDArray: _description_
    """
    if not all((isinstance(arg, np.ndarray) for arg in [field, y_flow, x_flow])):
        raise TypeError(
            f"Expected NDArray, got {type(field)}, {type(y_flow)}, {type(x_flow)}"
        )

    if not all(arg.ndim == 2 for arg in [field, y_flow, x_flow]):
        raise ValueError("Expected 2D input fields for all args")

    # TODO: check equal input shapes

    advected_field = np.zeros_like(field)

    # Loop over all features (non-zero elements) in field and advect by mean flow across the feature
    # Background field is assumed to be 0
    for feature_id in range(1, np.max(field) + 1):
        feature_mask = np.where(field == feature_id)
        dy = np.mean(y_flow[feature_mask]).astype(int)
        dx = np.mean(x_flow[feature_mask]).astype(int)

        # Now, advect the feature to the new position
        for y_coord, x_coord in zip(*feature_mask):
            advected_y_coord = y_coord + dy
            advected_x_coord = x_coord + dx

            # If this coordinate is out of bounds of the field, no further action needed
            oob_y_check = advected_y_coord < 0 or advected_y_coord > field.shape[0] - 1
            oob_x_check = advected_y_coord < 0 or advected_x_coord > field.shape[1] - 1
            if oob_y_check or oob_x_check:
                continue

            # If there is no label already at this position, can go ahead and place this feature
            # label here.
            id_at_coord = advected_field[advected_y_coord, advected_x_coord]
            if id_at_coord == 0:
                advected_field[advected_y_coord, advected_x_coord] = feature_id
                continue

            # Otherwise, need to handle conflicting features. Do this by finding distances between
            # the advected coordinate and the centroids of existing and iterating ids.
            # Choose the feature that is closest to its centroid
            existing_id_centroid = get_centroid(field, id_at_coord)
            iterating_id_centroid = get_centroid(field, feature_id)
            advected_coords = np.array([advected_y_coord, advected_x_coord])
            existing_id_centroid_distance = np.linalg.norm(
                existing_id_centroid - advected_coords
            )
            iterating_id_centroid_distance = np.linalg.norm(
                iterating_id_centroid - advected_coords
            )

            if iterating_id_centroid_distance < existing_id_centroid_distance:
                advected_field[advected_y_coord, advected_x_coord] = feature_id

    return advected_field


def generate_radial_mask(field: NDArray, coord: NDArray, mask_radius: int) -> NDArray:
    """
    Creates a radial mask of the same shape as the input field, centered on the (y,x) coord
    with radius equal to the mask radius.

    Args:
        field (NDArray):
            Field to generate mask for. Output will be the same shape
        coord (NDArray):
            Coordinate to centre the mask on
        mask_radius (int):
            Radius of values to include in circular mask.

    Returns:
        NDArray: Mask of values.
    """
    temp_y = np.arange(field.shape[0])
    temp_x = np.arange(field.shape[1])

    y_centroid_dist = (temp_y[:, np.newaxis] - coord[0]) ** 2
    x_centroid_dist = (temp_x[np.newaxis, :] - coord[1]) ** 2
    mask = (y_centroid_dist + x_centroid_dist) < mask_radius**2
    return mask


# TODO: replace this with the centroid calculation function in Feature?
def get_centroid(field: NDArray, value: int) -> NDArray:
    """
    From an input field, get the centroid location of a value of contiguous data

    Args:
        field (NDArray): _description_
        value (int): _description_

    Returns:
        NDArray: _description_
    """

    if not isinstance(field, np.ndarray):
        raise TypeError(f"Expected NDArray, got {type(field)}")

    if not np.issubdtype(type(value), np.integer):
        raise TypeError(f"Expected int, got {type(value)}")

    if not field.ndim == 2:
        raise ValueError("Expected 2D input fields")

    value_coords = np.where(field == value)
    centroid = np.mean(value_coords, axis=1)
    return centroid
