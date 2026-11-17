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
