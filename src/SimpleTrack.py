"""
Run the SimpleTrack algorithm to track objects through a sequence of images
"""

import sys
from yaml import safe_load
from pathlib import Path
import multiprocessing as mp

from Frame import Timeline, Frame
from FrameOutputManager import FrameOutputManager
from FrameTracker import FrameTracker
from OpticalFlowSolver import OpticalFlowSolver
from LoadingBar import LoadingBar


class SimpleTrack:
    def __init__(self, config_path: str):
        """
        Initialize SimpleTrack with configuration file

        Args:
            config_path (str):
                Path to the configuration file
        """
        with open(config_path, "r") as input:
            self.config = safe_load(input)
        self.start_time = self.config["DATETIME"]["start_time"]
        # TODO: make this optional: data might be passed in from external source
        self.filenames = self.__get_files_from_input_path(self.config["PATH"]["data"])
        self.timeline = Timeline()
        self.of_solver = OpticalFlowSolver(**self.config["OF_SOLVER"])
        self.frame_tracker = FrameTracker(**self.config["TRACKING"])
        self.frame_output = FrameOutputManager(
            self.config["PATH"]["output"],
            self.config["OUTPUT"]["experiment_name"],
            self.start_time,
            config_path,
        )

    def run(self, filenames=None):
        # If filesnames is provided, iterate only over these files.
        # This is useful for parallel processing.
        # Otherwise, iterate over all files in self.filenames
        if filenames is None:
            filenames = self.filenames

        self.loading_bar = LoadingBar(total=len(filenames))
        print(f"Hello from process {mp.current_process().name} with arg {filenames}\n")

        # Run the things
        for fnm_idx, filename in enumerate(filenames):
            frame = Frame()
            # TODO: change this procedure to a Loader class instead.
            # TODO: but, also want to offer a BasicLoader that can be interacted
            # with purely through the config file
            frame.load_mwe_data(filename)
            # frame.load_data(filename)
            frame.identify_features(**self.config["FEATURE"])
            self.timeline.add_to_timelime(frame)

            # If this is the first frame, skip tracking
            if len(self.timeline.timeline) == 1:
                # print(frame.get_time())
                # print(frame.get_features())
                # Output frame data to text file
                self.frame_output.features_to_txt(frame)
                self.frame_output.fields_to_npy(frame)
                continue

            # Now run optical flow between previous and current event
            prev_frame = self.timeline.get_previous_frame(frame.get_time())
            # Set max id for assigning to new features
            frame.set_max_id(prev_frame.get_max_id())
            y_flow, x_flow = self.of_solver.analyse_flow(prev_frame, frame)

            # Update the previous Frame with these displacements which is
            # needed for tracking Features
            # TODO: is the feautre assignment here actually needed?
            # TODO: the dy, d
            prev_frame.assign_displacements(y_flow, x_flow)

            # Track Features between difference Frames
            # print(frame.get_time())
            self.frame_tracker.run(prev_frame, frame)

            # Output frame data to text file
            self.frame_output.features_to_txt(frame)
            self.frame_output.fields_to_npy(frame)

            self.loading_bar.update_progress(fnm_idx + 1)
            # print("Final ids")
            # print(frame.get_features())

    def run_cset(self, time_and_data_dict: dict):
        # Run the things
        for time, data in time_and_data_dict.items():
            frame = Frame()
            frame.store_data(data, time)

            # frame.load_data(filename)
            frame.identify_features(**self.config["FEATURE"])
            self.timeline.add_to_timelime(frame)

            # If this is the first frame, skip tracking
            if len(self.timeline.timeline) == 1:
                print(frame.get_time())
                print(frame.get_features())
                continue

            # Now run optical flow between previous and current event
            prev_frame = self.timeline.get_previous_frame(frame.get_time())
            # Set max id for assigning to new features
            frame.set_max_id(prev_frame.get_max_id())
            y_flow, x_flow = self.of_solver.analyse_flow(prev_frame, frame)

            # Update the previous Frame with these displacements which is
            # needed for tracking Features
            # TODO:: is this actually needed??
            prev_frame.assign_displacements(y_flow, x_flow)

            # Track Features between difference Frames
            self.frame_tracker.run(prev_frame, frame)

            # self.loading_bar.update_progress(fnm_idx + 1)
            print("Final ids")
            print(frame.get_features())

    def run_parallel(self, processes=4):
        # Split filenames into chunks for each process
        chunk_size = len(self.filenames) // processes
        filename_chunks = [
            self.filenames[i : i + chunk_size]
            for i in range(0, len(self.filenames), chunk_size)
        ]

        with mp.Pool(processes=processes) as pool:
            pool.map(self.run, filename_chunks)

        # TODO: then need a way to make the results consistent between
        # different chunks.
        # I.e., if the last event of chunk 1 contains a storm that is
        # also present in the first event of chunk 2, then the chunk 2
        # storm needs to have a consistent ID, needs to have updated lifetimes
        # etc.
        # This is apparently already solved in Will Keats/Callum Scullion MO
        # code so don't need to reinvent the wheel here.

    def __get_files_from_input_path(self, input_path: str) -> list:
        supported_filetypes = [".nc", ".npy"]
        filenames = sorted(
            [
                p
                for p in Path(input_path).iterdir()
                if p.is_file() and p.suffix in supported_filetypes
            ]
        )
        return filenames


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Running SimpleTrack requires path to config")

    # For parallelisation, may need to setup the filenames here instead??

    config_path = sys.argv[1]
    SimpleTrack(config_path).run()
