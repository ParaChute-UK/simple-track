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

    def apply(self) -> Timeline:
        """
        Apply the filter to the given timeline and features.

        Returns:
        - A new Timeline containing only the filtered Features.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
                buffer is excluded.
                Defaults to 10.
        """
        super().__init__()
        self.boundary_buffer = boundary_buffer

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
        features_to_copy = []
        frame_times = list(timeline.get_timeline().keys())
        max_id = timeline.get_end_frame().max_id

        # Now, make a dict of
        # feature_id -> first_frame_time and then check if the feature is in the first frame


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
        if lifetime_window is None:
            lifetime_window = -1
        self.lifetime_window = lifetime_window


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
        if lifetime_window is None:
            lifetime_window = -1
        self.lifetime_window = lifetime_window


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
                The growth rate threshold to select features that reach this growth rate.
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
        if lifetime_window is None:
            lifetime_window = -1

        self.growth_rate = growth_rate
        self.centile = centile
        self.lifetime_window = lifetime_window


class MaximumFilter(BaseFilter):
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
        if lifetime_window is None:
            lifetime_window = -1
        self.lifetime_window = lifetime_window


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
        if lifetime_window is None:
            lifetime_window = -1
        self.lifetime_window = lifetime_window
