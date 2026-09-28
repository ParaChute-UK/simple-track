import numpy as np

from simpletrack.frame import Frame, Timeline
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
    1) Identifies whether the start and end frames of consectutive timelines are
    valid at the same time (this changes the tracking method in step 2).)
    2) Run FrameTracker on the end of the one timeline batch and the start of the
    next timeline batch to identify features that are tracked across the two timelines.
    Update lifetimes of these features accordingly.
    3) Update frame.max_id in the first frame of the second timeline batch to be the
    max_id of the last frame of the first timeline batch. Then, update any new feature
    ids in the second timeline batch to be unique by adding the max_id of the first
    timeline batch as an offset to the new feature ids in the second timeline batch.
    4) Search through the remaining frames in the batch to update:
        - tracked feature ids (reassigning them to the ids from the first frame,
        and updating lifetimes)
        - new feature ids (updating max_id for the frame, then updating any new feature
        ids to be unique using this max_id)
    5) Update all feature_field and lifetime_field with these
    """

    def __init__(self, timelines):
        self.timeline_batches = timelines

    def run(self) -> Timeline:
        """
        Stich together the input timelines to return a single timeline with
        consistent featyre IDs and fields across all frames

        Returns:
            Timeline: New Timeline object with consistent data across all frames
        """
