import itertools

from simpletrack.exceptions import SimpleTrackException
from simpletrack.flow_solver import FlowSolver
from simpletrack.frame import Frame, Timeline
from simpletrack.frame_output import LoadOutput
from simpletrack.frame_tracker import FrameTracker


class TimelineStitcher:
    """
    TimelineStitcher is a tool to create a continuous timeline from multiple separate
    runs of SimpleTrack. Rather than running tracking on a long series of data all at
    once, it may be necessary to split the analysis into batches.

    However, for separate timeline chunks, we run into two problems if we want to
    stitch them all together:
    1) Features that are not tracked across the end/start frames of consecutive
    timelines.
    2) New features that are identified in a later Frame will be given ids
    that are very likely to conflict with ids in an older Frame (as determined
    by frame.max_id property)

    To solve these problems, this tool performs the following steps:
    1) Run FrameTracker on the end of the one timeline batch and the start of the
    next timeline batch. This will identify features that are tracked across the two
    timelines and update lifetimes of these features accordingly. This will also
    identify unmatched features and assign an ID which will not conflict with existing
    IDs in the older frame.
    2) Search through the remaining frames in the newer batch to update:
        - tracked feature ids (reassigning them to the ids from the first frame,
        and updating lifetimes)
        - new feature ids (updating max_id for the frame, then updating any new feature
        ids to be unique using this max_id)
    3) Update all feature_field and lifetime_field with these properties
    """

    def __init__(
        self,
        timelines: list[Timeline | str],
        timeline_config: dict = None,
    ):
        """
        Run TimelineStitcher on a list of timelines to create a single timeline
        with consistent feature IDs and fields across all frames.

        Args:
            timelines (list(Timeline or str)):
                List of Timeline objects to be stitched together
                If inputs are Timelines, they will be used directly.
                If inputs are strings, they will be treated as paths to output data
                and will be loaded as Timeline objects using
                LoadOutput.load_to_timeline().
            timeline_config (dict, optional):
                One of the config files used to create the input Timeline objects.
                This is used to ensure that the same configuration settings
                are used when matching features between Frames in different timelines.
                If not provided, default settings will be used.

        """
        # Check input types
        if all(isinstance(t, str) for t in timelines):
            # If all inputs are strings, load them as Timeline objects
            loaded_timelines = []
            for t in timelines:
                try:
                    loaded_timeline = LoadOutput(t).load_to_timeline()
                    loaded_timelines.append(loaded_timeline)
                except Exception as e:
                    raise ValueError(f"Failed to load timeline from path {t}") from e
            timelines = loaded_timelines

        if not all(isinstance(t, Timeline) for t in timelines):
            raise TypeError("All input timelines must be of type Timeline")

        # Initialise the FlowSolver and FrameTracker with the provided config
        if timeline_config is not None:
            self.flow_solver = FlowSolver(**timeline_config["FLOW_SOLVER"])
            self.frame_tracker = FrameTracker(**timeline_config["TRACKING"])

            # Intitialise the retain_lifetime_on_split flag, which is used when
            # identifying features to propagate id information in
            # self.align_timeline_using_earliest_frame()
            # Default to True, as this is the default behaviour in FrameTracker
            self.retain_lifetime_on_split = timeline_config.get("TRACKING", {}).get(
                "retain_lifetime_on_split", True
            )
        else:
            self.flow_solver = FlowSolver()
            self.frame_tracker = FrameTracker()
            self.retain_lifetime_on_split = True
            print(
                "WARNING: No timeline_config provided. To ensure stitching is applied "
                "in a consistent way to the production of input Timelines, "
                "consider providing this argument."
            )

        # Sort by their start frame time
        self.timeline_batches = sorted(timelines, key=lambda t: t.start_time())

    def run(self) -> Timeline:
        """
        Stich together the input timelines to return a single timeline with
        consistent featyre IDs and fields across all frames

        Returns:
            Timeline: New Timeline object with consistent data across all frames
        """

        # Loop over all timeline pairs to match features between end of one and
        # start of the next
        for t1, t2 in itertools.pairwise(self.timeline_batches):
            print(
                f"Stitching timeline ending {t1.end_time()} "
                f"with timeline starting {t2.start_time()}"
            )

            # Assert t1, t2 are timelines
            if not isinstance(t1, Timeline) or not isinstance(t2, Timeline):
                raise TypeError("Both t1 and t2 must be of type Timeline")

            # Get the last frame of the first timeline and the first frame of the second
            older_frame = t1.get_end_frame()
            newer_frame = t2.get_start_frame()

            # Assert prev_frame, current_frame are frames
            if not isinstance(older_frame, Frame) or not isinstance(newer_frame, Frame):
                raise TypeError(
                    "Both prev_frame and current_frame must be of type Frame"
                )

            # Step 1) Align features between the two frames
            self.align_frames_at_timeline_seams(older_frame, newer_frame)

            # Step 2) Search through remaaining frames in the newer timeline
            # to update tracked feature ids, and make new feature ids unique
            # This will also promote the provisional IDs to final IDs in all
            # frames of the newer timeline
            self.align_timeline_using_earliest_frame(t2)

            # Step 3) Update the feature_field and lifetime_field in all frames of the
            # newer timeline to reflect the updated feature data
            # Finally, promote all provisional IDs to final IDs and
            # update feature_field and lifetime_field in all frames of the timeline
            for frame in t2.get_timeline().values():
                frame.update_fields_using_provisional_ids()
                frame.promote_provisional_ids()

        # Now, construct new Timeline object with all frames from all timelines
        stitched_frames = {}
        for timeline in self.timeline_batches:
            # This will overwrite any frames with the same timestamp
            stitched_frames.update(timeline.get_timeline())

        new_timeline = Timeline()
        for frame in stitched_frames.values():
            new_timeline.add_to_timelime(frame)

        return new_timeline

    def align_frames_at_timeline_seams(
        self, older_frame: Frame, newer_frame: Frame
    ) -> None:
        """
        Match features in the older frame with features in the newer frame.
        Update IDs in the newerframe to match those that were tracked from the
        older frame. Any untracked, new features in the newer timelineframe will
        be assigned new IDs that do not conflict with the older frame.

        Args:
            older_frame (Frame):
                Frame with older timestamps and features
            newer_frame (Frame):
                Frame with newer timestamps and features
        """

        # First, estimate flow between the two frames, to help with matching features
        y_flow, x_flow = self.flow_solver.analyse_flow(older_frame, newer_frame)

        # Update the current Frame with these displacements
        if y_flow is not None or x_flow is not None:
            newer_frame.assign_displacements(y_flow, x_flow)

        # First, make sure max_id is consistent between the two frames, so that
        # unmatched features in the newer frame will be assigned new IDs
        # that do not conflict
        if older_frame.max_id is not None:
            newer_frame.max_id = older_frame.max_id

        # Match features between frames
        # Using the dry_run flag means provisional IDs won't be promoted to final IDs,
        # meaning we can use these for propagating matching info throughout the timeline
        self.frame_tracker.run(older_frame, newer_frame, dry_run=True)

        # Now, in the new frame, any matched features will have their IDs updated
        # to match the older frame
        # Any unmatched features will have a new ID which does not condfluct with
        # the older frame
        # This will also have updated the lifetime properties of each feature in
        # the newer frame

    def align_timeline_using_earliest_frame(self, timeline: Timeline) -> None:
        """
        After frames have been aligned between two separate timelines, there will
        still be mismatches between features in the first frame (which we have
        aligned to the earlier timeline), and all other frames in this timeline

        This function iterates through all frames in the timeline and aligns them to
        the earliest frame by:
        1) Ensuring any features that are present in the earliest frame
        propagate their updated IDs to all subsequent frames in the timeline
        (using the provisional_id property of each feature to track this)
        2) Ensure any new features that are present in frames after the earliest
        frame are assigned new IDs that do not conflict with the earlier timelines
        (again using the provisional_id property to track this)

        Then, in each frame, promote the provisional IDs to final IDs now that
        full matching has been completed.

        Args:
            timeline (Timeline):
                Timeline with its earliest Frame aligned to an earlier Timeline
        """

        # First, populate the feature_map with old_id keys and new_id values
        # from the earliest frame
        feature_map = {}
        start_frame = timeline.get_start_frame()
        for feature in list(start_frame.features.values()):
            feature_map[feature.id] = feature.provisional_id

        # Iterate through the rest of the frames in the timeline and update their
        # feature IDs based on the feature_map
        all_frames = list(timeline.get_timeline().values())

        for prev_frame, frame in itertools.pairwise(all_frames):
            if not isinstance(frame, Frame) or not isinstance(prev_frame, Frame):
                raise TypeError("All frames in the timeline must be of type Frame")

            # Assign the new max_id to the current_frame
            frame.max_id = prev_frame.max_id

            for feature in list(frame.features.values()):
                # Check whether the feature is new by inspecting lifetime
                if feature.lifetime == 1:
                    # This is a new feature, assign it a new provisional ID that does
                    # not conflict with the previous frame
                    updated_id = frame.get_next_available_feature_id()
                    # Add this updated ID to the feature_map for future reference
                    feature_map[feature.id] = updated_id
                    feature.provisional_id = updated_id

                # Check if this feature has split from a parent and needs to
                # retain the lifetime of its parent
                elif self.retain_lifetime_on_split and feature.parent is not None:
                    # We also need to update the id of this feature, in the same
                    # way as the above condition
                    updated_id = frame.get_next_available_feature_id()
                    feature_map[feature.id] = updated_id
                    feature.provisional_id = updated_id

                    # Inherit the parent lifetime
                    feature.lifetime = frame.get_feature(feature.parent).lifetime

                # This feature was present in the earliest frame
                elif feature.id in feature_map:
                    # Update the feature's provisional ID and lifetime
                    updated_id = feature_map[feature.id]
                    feature.provisional_id = updated_id

                    # Get corresponding feature in the previous frame to update lifetime
                    prev_feature = prev_frame.get_feature(feature.id)
                    feature.lifetime = prev_feature.lifetime + 1

                else:
                    print(feature)
                    print(feature_map)
                    print(feature.parent)
                    msg = (
                        f"Feature with ID {feature.id} in frame {frame.time} does not "
                        "have a corresponding entry in the feature_map "
                        "and is not a new feature."
                    )
                    raise SimpleTrackException(msg)

        # We now need to loop through all Feature properties in all frames that use
        # Feature ID and update them using the feature map
        # Skip the first frame, as it is already aligned
        for frame in all_frames[1:]:
            self._update_feature_properties_using_feature_map(frame, feature_map)

    def _update_feature_properties_using_feature_map(
        self, frame: Frame, feature_map: dict
    ) -> None:
        """
        Update the properties of features in a frame using a feature map that
        maps old IDs to new IDs. This includes updating the parent, accreted,
        children, and accreted_in_next_frame_by properties of each feature.

        Args:
            frame (Frame): The frame whose features will be updated.
            feature_map (dict): A dictionary mapping old feature IDs to new feature IDs.
        """
        for feature in list(frame.features.values()):
            # Update parent ID if it exists
            if feature.parent is not None:
                feature.parent = feature_map[feature.parent]

            # Update accreted IDs if they exist
            if feature.accreted is not None:
                updated_accreted_list = []
                for accreted_id in feature.accreted:
                    if accreted_id in feature_map:
                        updated_accreted_list.append(feature_map[accreted_id])
                    else:
                        print(feature_map)
                        raise SimpleTrackException(
                            f"Accreted feature ID {accreted_id} in frame {frame.time} "
                            "does not have a corresponding entry in the feature_map."
                        )
                feature.accrete_ids(updated_accreted_list, replace=True)

            # Update children IDs if they exist
            if feature.children is not None:
                updated_children = []
                for child_id in feature.children:
                    if child_id in feature_map:
                        updated_children.append(feature_map[child_id])
                    else:
                        print(feature_map)
                        raise SimpleTrackException(
                            f"Child feature ID {child_id} in frame {frame.time} "
                            "does not have a corresponding entry in the feature_map."
                        )
                feature.spawns(updated_children, replace=True)

            # Update accreted_in_next_frame_by ID if it exists
            if feature.accreted_in_next_frame_by is not None:
                feature.accreted_in_next_frame_by = feature_map[
                    feature.accreted_in_next_frame_by
                ]
