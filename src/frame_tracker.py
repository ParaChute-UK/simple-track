import numpy as np
from numpy.typing import NDArray
from typing import Union
from frame import Frame
from feature import Feature
from utils import check_arrays


class FrameTracker:
    def __init__(
        self,
        overlap_nbhood: int = 5,
        overlap_threshold: float = 0.6,
        retain_lifetime_on_split: bool = True,
    ):
        """
        Initialise FrameTracker class to track Features between Frames

        Args:
            overlap_nbhood (int, optional):
                When calculating overlap between Features in advected and current Frames,
                code may apply a neighbourhood (nbhood) surrounding the Feature centroid
                if there is not a sufficient overlap found initilly. This value sets the
                radius of this nbhood in pixels.
                Defaults to 5.
            overlap_threshold (float, optional):
                Sets the minimum normalised overlap required between Features in advected
                and current Frames to be considered a match.
                Defaults to 0.6.
            retain_lifetime_on_split (bool, optional):
                If a child Feature splits from its parent feature, this determines whether
                the child Feature should carry over the lifetime from the parent or whether
                its lifetime should be set to 1
                Defaults to True
        """
        self.overlap_nbhood = int(overlap_nbhood)
        self.overlap_threshold = overlap_threshold
        self.retain_lifetime_on_split = retain_lifetime_on_split

    def run(self, prev_frame: Frame, current_frame: Frame) -> None:
        """
        Runs through the full Frame tracking procedure between two inputs.
        Step 1: Artifically advect features in the previous frame using its flow field. This will
        provide a best guess of where the features in the previous Frame should be located at the
        current timestep.

        Step 2: Match features between the advected frame and the current frame by assigning a
        new, proviosonal id to each Feature in the current Frame based on overlap with the advected
        Frame. Matched features will provisionally inherit the id and lifetime from the advected Frame.
        Also determine any accreted features during this matching. Any unmatched features
        are designated as new features and assigned a new id.

        Step 3: Check accreted ids from frame matching are not also present as provisional ids.
        Accreted ids should be removed from the field. If any are present, remove them from the accreted list.

        Step 4: After Feature matching, there may be multiple Features in current Frame that were matched
        to the same previous feature. These features will now have the same provisional ids.
        Find these ids and distinguish the most appropriate match (to retain the id) from the other matches
        (which will be designated as children and given a new id.)

        Step 5: Now that there is self consistent matched data in the current frame, use this to produce
        updated feature and lifetime fields in the current Frame using the provisional ids.

        Step 6: Promote provisional ids to final ids in current frame

        Step 7: Identify Features in the previous Frame that aren't matched with a Feature in the
        current Frame. This is useful for output statistics

        Args:
            prev_frame (Frame):
                Frame containing Features at the previous timestep
            current_frame (Frame):
                Frame containing Features and flow field at the current timestep

        Raises:
            TypeError: _description_
        """
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
        self.match_advected_and_current_frame_features(
            advected_frame, current_frame, prev_frame
        )

        # Step 3: Check accreted ids from frame matching are not also present as provisional ids.
        # Remove any accreted ids found as a provisional id in current frame
        self.check_accreted_feature_ids_are_not_provisional_ids(current_frame)

        # Step 4: After Feature matching, there may be multiple Features in current Frame that were
        # matched to the same previous feature. Resolve these conflicts
        self.resolve_provisional_id_conflicts(advected_frame, current_frame)

        # Step 5: Now that there is self consistent data in current frame, use this to
        # produce updated fields
        current_frame.update_fields_using_provisional_ids()

        # Step 6: Promote provisional ids to final ids in current frame
        current_frame.promote_provisional_ids()

        # Step 7: For tracing Features in the previous Frame that aren't matched with a
        # Feature in the current Frame. This is useful for output statistics
        self.identify_unmatched_features_in_prev_frame(prev_frame, current_frame)

    def identify_unmatched_features_in_prev_frame(
        self, prev_frame: Frame, current_frame: Frame
    ) -> None:
        """
        Identify Features in the previous Frame that are not matched with a Feature in the current Frame.
        This is useful for output statistics, e.g., for tracing dissipation events.
        Any feature that is not matched is designated as a final timestep by setting the final_timestep
        property to True.

        Args:
            prev_frame (Frame): Frame containing Features at previous timestep
            current_frame (Frame): Frame containing Features at current timestep
        """
        if not isinstance(prev_frame, Frame) or isinstance(current_frame, Frame):
            raise TypeError("Expected type Frame for both prev_frame and current_frame")

        current_frame_ids = [
            feature.id for feature in current_frame.get_features().values()
        ]
        for feature_id, feature in prev_frame.get_features().items():
            if feature_id not in current_frame_ids:
                feature.set_as_final_timestep()

    def resolve_provisional_id_conflicts(
        self, advected_frame: Frame, current_frame: Frame
    ) -> None:
        """
        After Feature matching, there may be multiple Features in current Frame that were
        matched to the same previous feature. These features will now have the same provisional ids.
        Find these ids and distinguish the most appropriate match (to retain the id) from the other matches
        (which will be designated as children and given a new id.)

        Matched "Parent" Features determined by largest overlap, and will retain the provisional id.
        Any other child Features will be assigned a new provisional id and have their parent attribute set
        to the retained provisional id. The parent Feature will have its children attribute updated to include
        the new child Feature ids.

        Args:
            advected_frame (Frame):
                Frame containing advected Features from previous timestep
            current_frame (Frame):
                Frame containing Features at current timestep

        """
        # First, list all provisional ids
        all_features = current_frame.get_features().values()
        all_provisional_ids = [feature.provisional_id for feature in all_features]

        # Find all provisional ids that are repeated
        unique_ids, counts = np.unique(all_provisional_ids, return_counts=True)
        conflicting_ids = unique_ids[counts > 1]

        # Loop over all Features with repeated provisional ids and designate parent/child
        for conflicting_id in conflicting_ids:
            # Find all Features with this provisional id
            matching_features = [
                feature
                for feature in all_features
                if feature.provisional_id == conflicting_id
            ]

            if not all(isinstance(feature, Feature) for feature in matching_features):
                raise TypeError("Expected all matching features to be of type Feature")

            # Get parent and child features
            parent_feature, child_features = self.identify_parent_and_child_features(
                conflicting_id,
                matching_features,
                advected_frame.get_feature_field(),
                current_frame.get_feature_field(),
            )

            # TODO: should some of this functionality be moved to Feature?
            # Preserve provisional id for the parent feature
            # All child features need new ids and are assigned the conflicting id as parent
            for feature in child_features:
                feature.parent = conflicting_id
                feature.provisional_id = current_frame.get_next_available_feature_id()
                # Handle lifetime depending on init input
                if self.retain_lifetime_on_split:
                    feature.lifetime = parent_feature.lifetime
                else:
                    feature.lifetime = 1

            # Update parent feature to include child ids
            parent_feature.children = [
                feature.provisional_id for feature in child_features
            ]

    def identify_parent_and_child_features(
        self,
        parent_id: int,
        matching_features: list[Feature],
        advected_feature_field: NDArray,
        current_feature_field: NDArray,
    ) -> list[Feature, list[Feature]]:
        """
        For a given target parent_id and list of matching Features (that all share this
        provisional parent_id), identify which Feature is the best match to be the parent.
        All others are identified as children.

        Best match is determined by finding the Feature that has the largest overlap with
        the advected feature with id parent_id.

        If multiple Features share the same overlap size, then the feature with the closest
        centroid is chosen.

        If multiple features are equidistant, the feature with the lower id is chosen.

        Args:
            parent_id (int):
                Provisional feature id to identify parent for
            matching_features (list[Feature]):
                List of matching Features sharing the provisional parent_id
            advected_frame (Frame):
                Frame containing advected Features from previous timestep
            current_frame (Frame):
                Frame containing Features at current timestep

        Returns:
            list[Feature, list[Feature]]:
                [parent_feature, list of child_features]
        """
        # First, check the parent_id is present in the advected feature field
        if not np.isin(parent_id, advected_feature_field):
            raise ValueError(
                f"Parent id {parent_id} not found in advected feature field"
            )

        # Find the feature that has the largest overlap with the advected feature
        # For this, need to match the locations of the repeated "provisional_id"
        # from the advetced feature field with the unique "id" in the current x
        # feature field
        overlap_sizes = []
        for feature in matching_features:
            advected_feature_mask = advected_feature_field == parent_id
            current_feature_mask = current_feature_field == feature.id
            overlap_size = self._number_of_overlapping_pixels(
                advected_feature_mask, current_feature_mask, parent_id, feature.id
            )
            overlap_sizes.append(overlap_size)

        # If there is no overlap between the two fields, implies there was a halo used
        # to match these features. Try applying the halo again here
        # TODO: think about a more rigorous way of deciding whether halo is needed here.
        if all(size == 0 for size in overlap_sizes):
            overlap_sizes = []
            for feature in matching_features:
                advected_feature_mask += generate_radial_mask(
                    advected_feature_field,
                    get_centroid(advected_feature_field, parent_id),
                    self.overlap_nbhood * np.count_nonzero(advected_feature_mask),
                )
                current_feature_mask += generate_radial_mask(
                    current_feature_field,
                    get_centroid(current_feature_field, feature.id),
                    self.overlap_nbhood * np.count_nonzero(current_feature_mask),
                )
                overlap_size = self._number_of_overlapping_pixels(
                    advected_feature_mask, current_feature_mask, parent_id, feature.id
                )
                overlap_sizes.append(overlap_size)

        if all(size == 0 for size in overlap_sizes):
            raise ValueError(
                f"No overlapping features found for provisional id {parent_id}"
            )

        # Pop feature with max overlap to be the parent feature,
        # If multiple features share max overlap, first instance is chosen.
        max_overlap_idx = np.argmax(overlap_sizes)
        parent_feature = matching_features.pop(max_overlap_idx)

        # TODO: if there is still more than one max overlap, check centroid!
        # The remaining features are the child features
        return parent_feature, matching_features

    def _number_of_overlapping_pixels(
        self, region1: NDArray, region2: NDArray, region1_id: int, region2_id: int
    ) -> int:
        return np.size(np.where((region1 == region1_id) & (region2 == region2_id)), 1)

    def check_accreted_feature_ids_are_not_provisional_ids(self, frame: Frame) -> None:
        """
        Any features that have been accreted by another feature should not therefore appear
        as a provisional id in the feature field (since it should no longer exist). This method
        checks that this is the case for all accreted ids. If any accreted ids are found to
        still exist as a provisional id, the accreted id is removed from its respective Feature

        Args:
            frame (Frame): Frame to inspect accreted ids
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"Expected type Frame, got {type(frame)}")

        all_features = frame.get_features().values()
        all_provisional_ids = [feature.provisional_id for feature in all_features]

        # Check each feature for accreted values
        for feature in all_features:
            if feature.accreted is None:
                continue
            if not isinstance(feature.accreted, list):
                raise TypeError(f"Expected list, got f{type(feature.accreted)}")

            # Copy accreted id to new list if it is not a provisional id
            new_accreted_list = [
                acc_id
                for acc_id in feature.accreted
                if acc_id not in all_provisional_ids
            ]
            # Reset the accreted feature id list
            # If list is empty, gets replaced with None in accreted setter
            feature.accreted = new_accreted_list

    def match_advected_and_current_frame_features(
        self, advected_frame: Frame, current_frame: Frame, prev_frame: Frame
    ) -> None:
        """
        For each Feature in the current Frame, attempt to match it to a Feature in
        the advected Frame by calcualating the overlap between the two fields.

        Matched Features are assigned to the provisional id property of the current
        Feature and inherit the lifetime from the advected Feature.

        Other Features in the advected field that contain a sufficient overlap but
        that are not the best match are added to the accreted property of the current Feature.

        Unmatched Features in the current Frame are assigned a new provisional id.

        Args:
            advected_frame (Frame):
                Frame containing advected Features from previous timestep
            current_frame (Frame):
                Frame containing Features at current timestep
        """
        # Get the feature fields to analyse
        advected_feature_field = advected_frame.get_feature_field()
        current_feature_field = current_frame.get_feature_field()

        # Attempt to match features in the advected frame with current frame
        for current_feature in current_frame.get_features().values():
            if not isinstance(current_feature, Feature):
                raise TypeError(f"Expected Feature, got {type(current_feature)}")

            feature_id = current_feature.id

            # Count the ids contained in advected_feature_field that are in the same
            # position as the feature_id in the current_feature_field (normalised by
            # the feature sizes)
            overlap_hist = self.calculate_overlap_histogram(
                advected_feature_field, current_feature_field, feature_id, nbhood=0
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
                matching_id = current_frame.get_next_available_feature_id()
                current_feature.lifetime = 1
            else:
                # Inherit lifetime from matching feature
                matching_feature = advected_frame.get_feature(matching_id)
                current_feature.lifetime = matching_feature.lifetime + 1

            # Provisionally assign the matching_id to this feature
            current_feature.provisional_id = matching_id

            if other_sufficient_ids is not None:
                # Add other ids to Feature accretion list
                current_feature.accrete_ids(other_sufficient_ids)

                # Update the accreted_in_next_frame_by property of Features in prev_frame
                for accreted_id in other_sufficient_ids:
                    accreted_feature = prev_frame.get_feature(accreted_id)
                    accreted_feature.accreted_in_next_frame_by = feature_id
                    accreted_feature.set_as_final_timestep()

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

        - If there is more than one sufficient overlap, find the Feature label with the closest size
        If there is more than one Feature shares a closest size, find the Feature from these that is
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
        # Check number of sufficient overlaps. Get bool array of values meeting threshold
        sufficient_overlaps = overlap_hist >= self.overlap_threshold
        len_sufficient_overlaps = np.count_nonzero(sufficient_overlaps)

        # Setup returning variable that will only be not None if there are multiple overlaps
        other_sufficient_ids = None

        if len_sufficient_overlaps == 0:
            matching_id = None

        if len_sufficient_overlaps == 1:
            matching_id = np.argmax(overlap_hist)

        # If there is more than one sufficient overlap, keep the properties of the feature
        # with the closest size. If multiple have overlaps, keep nearest in centroid
        if len_sufficient_overlaps > 1:
            # Check for size of each feature in advected_frame with sufficient overlap
            ids_of_sufficient_overlaps = np.argwhere(sufficient_overlaps).squeeze()
            min_size_comparison = self.find_ids_of_closest_size(
                field_with_id=current_feature_field,
                field_to_search=advected_feature_field,
                target_feature_id=current_feature_id,
                candidate_ids=ids_of_sufficient_overlaps.tolist(),
            )

            # If only one id has a closest size to target feature, this is the matching id
            if len(min_size_comparison) == 1:
                matching_id = min_size_comparison[0]

            # If more than one id shares a closest size, find the closest centroid
            else:
                # Get the closest centroid for each feature sharing a minimum distance
                centroid_distances = {}
                current_feature_centroid = get_centroid(
                    current_feature_field, current_feature_id
                )
                for overlap_id in min_size_comparison:
                    overlap_id_centroid = get_centroid(
                        advected_feature_field, overlap_id
                    )
                    distance = np.linalg.norm(
                        current_feature_centroid - overlap_id_centroid
                    )
                    centroid_distances[overlap_id] = distance

                # Find the feature that has the minimum distance. If there are still more
                # than 1 possible options at this stage, min returns the first instance
                matching_id = min(centroid_distances, key=centroid_distances.get)

            # Add the other sufficient overlaps to other_sufficient_ids
            # To ensure the matching id is now not included in other sufficient ids,
            # set sufficient overlaps to False at this id
            sufficient_overlaps[matching_id] = False
            other_sufficient_ids = np.argwhere(sufficient_overlaps)
            # Squeeze output, but only one axis so that single element arrays remain arrays
            axis_to_squeeze = other_sufficient_ids.shape.index(1)
            other_sufficient_ids = other_sufficient_ids.squeeze(axis_to_squeeze)

        return matching_id, other_sufficient_ids

    def find_ids_of_closest_size(
        self,
        field_with_id: NDArray,
        field_to_search: NDArray,
        target_feature_id: int,
        candidate_ids: list[int],
    ) -> list[int]:
        """
        Given a list of candidate ids, finds the id whose size is closest to the size
        of the feature with feature_id in field_with_id

        Args:
            field_with_id (NDArray):
                Feature field containing the feature with feature_id
            field_to_search (NDArray):
                Feature field containing the candidate ids
            feature_id (int):
                Id of the feature in field_with_id to compare sizes against
            candidate_ids (list):
                List of candidate ids in field_to_search to compare sizes against
        Returns:
            list[int]:
                List of candidate ids that have the closest size to the target feature
        """
        field_with_id, field_to_search = check_arrays(
            field_with_id,
            field_to_search,
            ndim=2,
            equal_shape=True,
            dtype=int,
        )

        size_of_target_feature_in_target_field = np.size(
            np.where(field_with_id == target_feature_id), 1
        )
        size_of_candidate_features = {
            candidate_id: np.size(np.where(field_to_search == candidate_id), 1)
            for candidate_id in candidate_ids
        }
        size_comparison = {
            candidate_id: np.abs(size_of_target_feature_in_target_field - size)
            for candidate_id, size in size_of_candidate_features.items()
        }
        min_size_comparison = np.min(list(size_comparison.values()))
        closest_size_ids = [
            candidate_id
            for candidate_id, size_diff in size_comparison.items()
            if size_diff == min_size_comparison
        ]
        return closest_size_ids

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
        in this reigon in the adveced feature field. This is normalised by the pixel size of each
        Feature in the advected field to get the degree of overlap.

        Args:
            advected_feature_field (NDArray):
                Field containing advected features
            current_feature_field (NDArray):
                Field containing current features
            feature_id (int):
                Feature ID in the current feature field to calculate overlap for
            nbhood (int, optional):
                If nonzero, applied a radial mask surrouding the feature centroid
                of the current_feature_field to expand the overlap calculation.
                Defaults to 0.

        Returns:
            NDArray: Array contanining normalised overlap values for each feature id in
            the advected feature field
        """
        advected_feature_field, current_feature_field = check_arrays(
            advected_feature_field,
            current_feature_field,
            ndim=2,
            equal_shape=True,
            dtype=int,
        )
        if not isinstance(feature_id, (int, np.integer)):
            raise TypeError(f"Expected int, got {type(feature_id)}")
        if not isinstance(nbhood, (int, np.integer)):
            raise TypeError(f"Expected int, got {type(nbhood)}")
        if nbhood < 0:
            raise ValueError(f"Expected non-negative nbhood, got {nbhood}")

        # Create feature mask using current feature field, or expand mask using a nbhood
        # if this is flagged in input
        feature_mask = np.where(current_feature_field == feature_id, True, False)
        if nbhood:
            centroid = get_centroid(current_feature_field, feature_id)
            radial_mask_size = nbhood * np.count_nonzero(feature_mask)
            feature_mask += generate_radial_mask(
                current_feature_field, centroid, radial_mask_size
            )

        # Setup bins for comparing feature fields using histogram
        # Need to find max value among both input fields
        input_fields = [advected_feature_field, current_feature_field]
        max_val = np.max(input_fields)
        bins = np.arange(int(max_val) + 2)

        # Find overlap between two feature fields by finding histogram of points
        # using mask of current features applied to the advected feature field
        overlap_hist = np.histogram(advected_feature_field[feature_mask], bins)[0]

        # Set the first value of the hist to 0 since this represents the background
        overlap_hist[0] = 0

        # Normalise overlap histogram by size of each feature in advected field only
        norm_sizes = np.array(
            [
                np.count_nonzero(advected_feature_field == idx)
                for idx in range(len(overlap_hist))
            ]
        )
        # Replace any zero sizes with 1 to avoid division by zero
        norm_sizes = np.where(norm_sizes == 0, 1, norm_sizes)
        overlap_normed = overlap_hist / norm_sizes
        return overlap_normed

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

        # If there is no flow field, return the un-advected frame
        y_flow, x_flow = frame.get_flow()
        if y_flow is None or x_flow is None:
            print(f"y_flow: {y_flow}")
            print(f"x_flow: {x_flow}")
            print("Continuing with unadvected Frame")
            return frame

        feature_field = frame.get_feature_field()
        advected_feature_field = advect_field_using_motion_vectors(
            feature_field, y_flow, x_flow
        )

        advected_frame = Frame()
        advected_frame.set_feature_field(advected_feature_field)
        advected_frame.populate_features()

        # Transfer lifetimes to advected frame
        for advected_feature in advected_frame.get_features().values():
            advected_id = advected_feature.id
            advected_feature.lifetime = frame.get_feature(advected_id).lifetime

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
        field (NDArray):
            Field containing feature labels to be advected. Assumes 0 is the background value
        y_flow (NDArray):
            2D array of same shape as field containing y motion vectors
        x_flow (NDArray):
            2D array of same shape as field containing x motion vectors

    Returns:
        NDArray:
            Advected field
    """
    field, y_flow, x_flow = check_arrays(
        field, y_flow, x_flow, ndim=2, equal_shape=True
    )

    advected_field = np.zeros_like(field)

    # Loop over all features (non-zero elements) in field and advect by mean flow across the feature
    # Background field is assumed to be 0
    for feature_id in range(1, np.max(field) + 1):
        feature_mask = np.where(field == feature_id)
        # If mask is empty, this Feature is not in the current field
        if np.size(feature_mask) == 0:
            continue

        # For the purposes of advecting features, need dy, dx to align with grid points
        # Therefore, perform integer mean with rounding if dy, dx values are floats
        dy = np.mean(y_flow[feature_mask], dtype=int)
        dx = np.mean(x_flow[feature_mask], dtype=int)

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
            # TODO: should this not merge features instead??
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


def get_centroid(field: NDArray, value: int) -> NDArray:
    """
    From an input field, get the centroid location of a value of contiguous data

    Args:
        field (NDArray):
            Field containing feature to find centroid for
        value (int):
            Value in field to find centroid of

    Returns:
        NDArray:
            (y, x) centroid
    """

    field = check_arrays(field, ndim=2)

    if not np.issubdtype(type(value), np.integer):
        raise TypeError(f"Expected int, got {type(value)}")

    value_coords = np.where(field == value)
    centroid = np.mean(value_coords, axis=1)
    return centroid
