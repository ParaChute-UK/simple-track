import numpy as np

from simpletrack.feature import Feature
from simpletrack.frame import Frame, Timeline


class BaseFilter:
    """
    Base class for filters. All filters should inherit from this class and implement
    the `apply` method.

    Filtering takes an input Timeline and a selection of Features and returns a new
    Timeline that only includes the filtered Features.
    """

    def __init__(self):
        pass

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that match the filter criteria. This method should be
        implemented by subclasses.

        Returns:
        - A list of IDs that match the filter criteria.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def identify_centile_threshold(
        self, feature_info_dict: dict, centile: float, lifetime_window: int = None
    ) -> float:
        """
        Identify the threshold value for a given centile from a list of values.
        If lifetime_window is specified, only the first `lifetime_window` values will be
        considered for each feature.

        Parameters:
            feature_info_dict:
                A dictionary containing feature information, where keys are
                feature IDs and values are lists of values for each feature.
            centile:
                The centile to calculate (0-100).
            lifetime_window:
                Optional parameter to limit the number of values considered
                for each feature. If specified, only the first `lifetime_window` values
                will be considered.

        Returns:
        - The threshold value corresponding to the specified centile.
        """
        all_vals = []
        for vals in feature_info_dict.values():
            if lifetime_window is not None:
                vals = vals[:lifetime_window]
            all_vals.extend(vals)
        return np.percentile(all_vals, centile)

    def _copy_features_to_new_timeline(
        self, src_timeline: Timeline, features_to_copy: list[Feature]
    ) -> Timeline:
        """
        Create a new Timeline and copy the specified Features from the source Timeline.

        Parameters:
        - src_timeline: The source Timeline from which to copy Features.
        - features_to_copy: A list of Features to copy to the new Timeline.

        Returns:
        - A new Timeline containing only the copied Features.
        """
        new_timeline = Timeline()

        # Create a new Frame for each Frame in the source Timeline
        # Copy over the specified Features and their associated data to the new Frame
        frame_times = list(src_timeline.timeline.keys())

        if len(frame_times) == 0:
            return new_timeline

        for frame_time in frame_times:
            new_frame = Frame()
            new_frame.time = frame_time
            src_frame = src_timeline.get_frame(frame_time)

            # Search each Frame in the source Timeline for any of the Features to copy
            frame_features_dict = src_frame.features
            filtered_features_dict = {
                id: frame_features_dict[id]
                for id in features_to_copy
                if id in frame_features_dict
            }
            new_frame.features = filtered_features_dict

            # Also create new fields to add to the new_frame
            new_raw_field = np.zeros_like(src_frame.raw_field)
            new_lifetime_field = np.zeros_like(src_frame.lifetime_field)
            new_feature_field = np.zeros_like(src_frame.feature_field)

            # Populate the new fields with the data
            features_mask = np.isin(src_frame.feature_field, features_to_copy)
            new_raw_field[features_mask] = src_frame.raw_field[features_mask]
            new_lifetime_field[features_mask] = src_frame.lifetime_field[features_mask]
            new_feature_field[features_mask] = src_frame.feature_field[features_mask]

            # Add to the frame
            new_frame.raw_field = new_raw_field
            new_frame.lifetime_field = new_lifetime_field
            new_frame.feature_field = new_feature_field

            # Also copy motion field if it exists
            flow = src_frame.get_flow()
            if flow[0] is not None and flow[1] is not None:
                new_frame.y_flow, new_frame.x_flow = flow

            # Add to timeline
            new_timeline.add_to_timelime(new_frame)

        return new_timeline

    def apply(self, timeline: Timeline) -> Timeline:
        """
        Apply the Filter to the given timeline and return a new timeline
        containing only the filtered Features.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            Timeline:
                A new Timeline containing only the filtered Features.
        """
        features_to_copy = self.get_ids_matching_criteria(timeline)
        filtered_timeline = self._copy_features_to_new_timeline(
            timeline, features_to_copy
        )
        return filtered_timeline


class InitFilter(BaseFilter):
    """
    Filter that selects Features that are newly initialised when they first appeared
    in the Timeline (excluding the first Frame)
    """

    def __init__(self, boundary_buffer: int = 10):
        """
        Initialise the InitFilter with a boundary buffer to exclude features that
        initiliase too close to the edge of the frame (and are therefore likely to
        have advected into the domain rather than being newly initialised).

        Args:
            boundary_buffer (int, optional):
                The boundary buffer to exclude features where any pixel within the
                buffer is excluded. If None, no boundary buffer is applied.
                Defaults to 10.
        """
        super().__init__()
        self.boundary_buffer = boundary_buffer

    def within_boundary(
        self, feature: Feature, frame: Frame, boundary_buffer: int
    ) -> bool:
        """
        Check if the Feature is inside the specified boundary of the Frame.

        Args:
            feature (Feature):
                The Feature to check.
            frame (Frame):
                The Frame containing the Feature.

        Returns:
            bool:
                True if the Feature is within the boundary, False otherwise.
        """
        if boundary_buffer is None:
            return True

        # Get the coordinates of the feature
        feature_y_coords, feature_x_coords = feature.coords

        # Calculate valid (safe) boundaries
        x_valid_coords = np.arange(
            boundary_buffer, frame.raw_field.shape[1] - boundary_buffer
        )
        y_valid_coords = np.arange(
            boundary_buffer, frame.raw_field.shape[0] - boundary_buffer
        )

        # Return True only if all feature coords are within the valid boundaries
        return all(np.isin(feature_x_coords, x_valid_coords)) and all(
            np.isin(feature_y_coords, y_valid_coords)
        )

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that are newly initialised in the Timeline.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            list[Feature]:
                A list of Features that are newly initialised in the Timeline.
        """
        frame_times = list(timeline.get_timeline().keys())

        # Now, make a dict of feature_id -> first_frame_time
        first_frame_time = {}
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id in frame.features:
                if feature_id not in first_frame_time:
                    first_frame_time[feature_id] = frame_time

        # Remove any features that are newly initialised in the first frame
        # Only copy ids that are initialised in subsequent frames to feature_to_copy
        ids_init_after_first_frame = []
        if not first_frame_time:
            return ids_init_after_first_frame

        for feature_id, init_time in first_frame_time.items():
            if init_time != frame_times[0]:
                ids_init_after_first_frame.append(feature_id)

        # Check which of the remaining features are newly initialised. Remove
        # any that don't meet the is_new() criteria or boundary criteria
        filtered_ids = []
        for feature_id in ids_init_after_first_frame:
            feature_time = first_frame_time[feature_id]
            frame = timeline.get_frame(feature_time)
            feature = frame.get_feature(feature_id)

            # Skip any features that are not newly initialised
            if not feature.is_new():
                continue

            # Skip any features that are too close to the boundary of the frame
            if not self.within_boundary(feature, frame, self.boundary_buffer):
                continue

            filtered_ids.append(feature_id)

        return filtered_ids


class NonInteractionFilter(BaseFilter):
    """
    Filter that selects Features that do not interact with any other Features in the
    Timeline. This is determined by checking if the Feature merges, splits or accretes
    other features at any point in the Timeline (or within first x
    frames of its lifetime)
    """

    def __init__(self, lifetime_window: int = None):
        """
        Initialise the NonInteractionFilter. If lifetime_window is specified,
        the filter will only consider interactions within the first x frames
        of the Feature's lifetime. Otherwise, the filter will consider interactions
        over the entire lifetime of the Feature (until it dissipates, merges into
        another feature, or advects out of the domain).

        Args:
            lifetime_window (int, optional):
                The number of frames to consider for interactions. If None, the entire
                lifetime of the Feature is considered.
                Defaults to None.
        """
        super().__init__()
        self.lifetime_window = lifetime_window

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that do not interact with any other Features in the
        Timeline.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            list[int]:
                A list of Features that do not interact with any other Features in the
                Timeline.
        """
        frame_times = list(timeline.get_timeline().keys())

        # Now, make a dict of feature_id -> first_frame_time
        first_frame_time = {}
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id in frame.features:
                if feature_id not in first_frame_time:
                    first_frame_time[feature_id] = frame_time

        # For each feature, check if it interacts with any other feature
        # in the timeline or within the lifetime_window.
        # If it does, remove it from the list of features to copy
        filtered_ids = []
        for feature_id, init_time in first_frame_time.items():
            # Get the final frame to search up to based on lifetime window
            if self.lifetime_window is None:
                final_time = frame_times[-1]
            else:
                init_time_idx = frame_times.index(init_time)
                final_time_idx = init_time_idx + self.lifetime_window
                if final_time_idx >= len(frame_times):
                    final_time_idx = len(frame_times) - 1
                final_time = frame_times[final_time_idx]

            # Check for interactions in the timeline between init_time and final_time
            frames_to_check = [
                frame_time
                for frame_time in frame_times
                if init_time <= frame_time <= final_time
            ]
            is_non_interacting = True
            for frame_time in frames_to_check:
                frame = timeline.get_frame(frame_time)
                feature = frame.get_feature(feature_id)
                if feature is None:
                    continue
                if feature.is_interacting():
                    is_non_interacting = False
                    break

            # Only add if non-interacting across all checked frames
            if is_non_interacting:
                filtered_ids.append(feature_id)

        return filtered_ids


class LifetimeFilter(BaseFilter):
    """
    Filter that selects Features that reach a certain lifetime at some stage
    in the Timeline
    """

    def __init__(self, lifetime: int = None, centile: float = None):
        """
        Initialise the LifetimeFilter with a lifetime threshold or centile to select
        features that reach a certain lifetime at some stage in the Timeline.

        Args:
            lifetime (int, optional):
                The lifetime threshold to select features that reach this lifetime.
                Defaults to None.
            centile (float, optional):
                The centile threshold to select features that reach this lifetime.
                Defaults to None.
        """
        super().__init__()
        if lifetime is None and centile is None:
            raise ValueError("Either lifetime or centile must be specified.")

        self.lifetime = lifetime
        self.centile = centile

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that reach a certain lifetime at some stage in the
        Timeline.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            list[int]:
                A list of Features that reach a certain lifetime at some stage in the
                Timeline.
        """
        frame_times = list(timeline.get_timeline().keys())
        feature_lifetimes = {}

        # Get feature lifetimes across all times
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id, feature in frame.features.items():
                if feature_id not in feature_lifetimes:
                    feature_lifetimes[feature_id] = []
                feature_lifetimes[feature_id].append(feature.lifetime)

        # Define the lifetime threshold based on the centile if specified, otherwise
        # use the lifetime threshold directly
        if self.centile is not None:
            lifetime_threshold = self.identify_centile_threshold(
                feature_lifetimes, self.centile
            )
        else:
            lifetime_threshold = self.lifetime

        # Get the IDs of features that meet the criteria
        filtered_ids = []
        for feature_id, lifetimes in feature_lifetimes.items():
            if np.any(np.array(lifetimes) >= lifetime_threshold):
                filtered_ids.append(feature_id)

        return filtered_ids


class SizeFilter(BaseFilter):
    """
    Filter that selects Features that reach a certain size at some stage
    in the Timeline
    """

    def __init__(
        self, size: int = None, centile: float = None, lifetime_window: int = None
    ):
        """
        Initialise the SizeFilter with a size threshold or centile to select
        features that reach a certain size at some stage in the Timeline.

        Args:
            size (int, optional):
                The size threshold to select features that reach this size.
                Defaults to None.
            centile (float, optional):
                The centile threshold to select features that reach this size.
                Value associated with the centile is calculated from the distribution
                of sizes of all features within their given lifetime window.
                Defaults to None.
            lifetime_window (int, optional):
                The number of frames to consider for size. If None, the entire
                lifetime of the Feature is considered.
                Defaults to None.
        """
        super().__init__()
        if size is None and centile is None:
            raise ValueError("Either size or centile must be specified.")

        self.size = size
        self.centile = centile
        self.lifetime_window = lifetime_window

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that reach a certain size at some stage in the
        Timeline.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            list[int]:
                A list of Features that reach a certain size at some stage in the
                Timeline.
        """
        frame_times = list(timeline.get_timeline().keys())
        feature_sizes = {}

        # Get feature sizes across all times
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id, feature in frame.features.items():
                if feature_id not in feature_sizes:
                    feature_sizes[feature_id] = []
                feature_sizes[feature_id].append(feature.get_size())

        # Define the size threshold based on the centile if specified, otherwise use the
        # size threshold directly
        if self.centile is not None:
            size_threshold = self.identify_centile_threshold(
                feature_sizes, self.centile, self.lifetime_window
            )
        else:
            size_threshold = self.size

        # Now, check which features meet the size threshold
        filtered_ids = []
        for feature_id, sizes in feature_sizes.items():
            if self.lifetime_window is not None:
                sizes = sizes[: self.lifetime_window]

            if np.any(np.array(sizes) >= size_threshold):
                filtered_ids.append(feature_id)

        return filtered_ids


class GrowthFilter(BaseFilter):
    """
    Filter that selects Features that reach a certain growth rate at some stage
    in the Timeline
    """

    def __init__(
        self,
        growth_rate: float = None,
        centile: float = None,
        lifetime_window: int = None,
    ):
        """
        Initialise the GrowthFilter with a growth rate threshold or centile to select
        features that reach a certain growth rate at some stage in the Timeline.

        Args:
            growth_rate (float, optional):
                The growth rate threshold to select features that reach this growth
                rate.
                Defaults to None.
            centile (float, optional):
                The centile threshold to select features that reach this growth rate.
                Defaults to None.
            lifetime_window (int, optional):
                The number of frames to consider for growth rate. If None, the entire
                lifetime of the Feature is considered.
                Defaults to None.
        """
        super().__init__()
        if growth_rate is None and centile is None:
            raise ValueError("Either growth_rate or centile must be specified.")

        self.growth_rate = growth_rate
        self.centile = centile
        self.lifetime_window = lifetime_window

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get the IDs of features that match the growth rate criteria.

        Args:
            timeline (Timeline): The Timeline containing the features to filter.

        Returns:
            list[int]: A list of feature IDs that match the criteria.
        """
        frame_times = list(timeline.get_timeline().keys())
        feature_growth_rates = {}

        # Get feature growth rates across all times
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id, feature in frame.features.items():
                if feature_id not in feature_growth_rates:
                    feature_growth_rates[feature_id] = []
                feature_growth_rates[feature_id].append(feature.growth)

        # Remove "None" values from initial growth rates for each feature
        for feature_id in feature_growth_rates:
            feature_growth_rates[feature_id] = feature_growth_rates[feature_id][1:]

        # Define the growth rate threshold based on the centile if specified, otherwise
        # use the growth rate threshold directly
        if self.centile is not None:
            growth_rate_threshold = self.identify_centile_threshold(
                feature_growth_rates, self.centile, self.lifetime_window
            )
        else:
            growth_rate_threshold = self.growth_rate

        # Now, check which features meet the growth rate threshold
        filtered_ids = []
        for feature_id, growth_rates in feature_growth_rates.items():
            if self.lifetime_window is not None:
                growth_rates = growth_rates[: self.lifetime_window]

            if np.any(np.array(growth_rates) >= growth_rate_threshold):
                filtered_ids.append(feature_id)

        return filtered_ids


class MaxFilter(BaseFilter):
    """
    Filter that selects Features that reach a certain maximum value at some stage
    in the Timeline
    """

    def __init__(
        self, maximum: float = None, centile: float = None, lifetime_window: int = None
    ):
        """
        Initialise the MaximumFilter with a maximum threshold or centile to select
        features that reach a certain maximum at some stage in the Timeline.

        Args:
            maximum (float, optional):
                The maximum threshold to select features that reach this maximum.
                Defaults to None.
            centile (float, optional):
                The centile threshold to select features that reach this maximum.
                Defaults to None.
            lifetime_window (int, optional):
                The number of frames to consider for maximum. If None, the entire
                lifetime of the Feature is considered.
                Defaults to None.
        """
        super().__init__()
        if maximum is None and centile is None:
            raise ValueError("Either maximum or centile must be specified.")

        self.maximum = maximum
        self.centile = centile
        self.lifetime_window = lifetime_window

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that reach a certain maximum at some stage in the
        Timeline.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            list[int]:
                A list of Features that reach a certain maximum at some stage in the
                Timeline.
        """
        frame_times = list(timeline.get_timeline().keys())
        feature_maxima = {}

        # Get feature maxima across all times
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id, feature in frame.features.items():
                if feature_id not in feature_maxima:
                    feature_maxima[feature_id] = []
                feature_maxima[feature_id].append(feature.max)

        # Define the maximum threshold based on the centile if specified, otherwise use
        # the maximum threshold directly
        if self.centile is not None:
            max_threshold = self.identify_centile_threshold(
                feature_maxima, self.centile, self.lifetime_window
            )
        else:
            max_threshold = self.maximum

        # Now, check which features meet the maximum threshold
        filtered_ids = []
        for feature_id, maxima in feature_maxima.items():
            if self.lifetime_window is not None:
                maxima = maxima[: self.lifetime_window]

            if np.any(np.array(maxima) >= max_threshold):
                filtered_ids.append(feature_id)

        return filtered_ids


class MeanFilter(BaseFilter):
    """
    Filter that selects Features that reach a certain mean value at some stage
    in the Timeline
    """

    def __init__(
        self, mean: float = None, centile: float = None, lifetime_window: int = None
    ):
        """
        Initialise the MeanFilter with a mean threshold or centile to select
        features that reach a certain mean at some stage in the Timeline.

        Args:
            mean (float, optional):
                The mean threshold to select features that reach this mean.
                Defaults to None.
            centile (float, optional):
                The centile threshold to select features that reach this mean.
                Defaults to None.
            lifetime_window (int, optional):
                The number of frames to consider for mean. If None, the entire
                lifetime of the Feature is considered.
                Defaults to None.
        """
        super().__init__()
        if mean is None and centile is None:
            raise ValueError("Either mean or centile must be specified.")

        self.mean = mean
        self.centile = centile
        self.lifetime_window = lifetime_window

    def get_ids_matching_criteria(self, timeline: Timeline) -> list[int]:
        """
        Get a list of Features that reach a certain mean at some stage in the
        Timeline.

        Args:
            timeline (Timeline):
                The Timeline to filter.

        Returns:
            list[int]:
                A list of Features that reach a certain mean at some stage in the
                Timeline.
        """
        frame_times = list(timeline.get_timeline().keys())
        feature_means = {}

        # Get feature means across all times
        for frame_time in frame_times:
            frame = timeline.get_frame(frame_time)
            for feature_id, feature in frame.features.items():
                if feature_id not in feature_means:
                    feature_means[feature_id] = []
                feature_means[feature_id].append(feature.mean)

        # Define the mean threshold based on the centile if specified, otherwise use the
        # mean threshold directly
        if self.centile is not None:
            mean_threshold = self.identify_centile_threshold(
                feature_means, self.centile, self.lifetime_window
            )
        else:
            mean_threshold = self.mean

        # Now, check which features meet the mean threshold
        filtered_ids = []
        for feature_id, means in feature_means.items():
            if self.lifetime_window is not None:
                means = means[: self.lifetime_window]

            if np.any(np.array(means) >= mean_threshold):
                filtered_ids.append(feature_id)

        return filtered_ids
