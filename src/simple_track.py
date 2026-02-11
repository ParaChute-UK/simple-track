"""
Run the SimpleTrack algorithm to track objects through a sequence of images
"""

import sys
from yaml import safe_load
from pathlib import Path
import multiprocessing as mp

from frame import Timeline, Frame
from frame_output import FrameOutputManager
from frame_tracker import FrameTracker
from flow_solver import OpticalFlowSolver
from load import LoadingBar, get_loader


class SimpleTrack:
    def __init__(self, config_path: str):
        """
        Initialize SimpleTrack with configuration file

        Args:
            config_path (str):
                Path to the configuration file
        """
        self.config = self._read_config(config_path)
        self.start_time = self.config["DATETIME"]["start_time"]
        # TODO: make this optional: data might be passed in from external source
        self.filenames = self.__get_files_from_input_path(self.config["PATH"]["data"])
        self.loader = get_loader(self.config["PATH"]["loader"])
        self.timeline = Timeline()
        self.of_solver = OpticalFlowSolver(**self.config["OF_SOLVER"])
        self.frame_tracker = FrameTracker(**self.config["TRACKING"])

        if "output" not in self.config["PATH"].keys():
            output_path = "./output"
        else:
            output_path = self.config["PATH"]["output"]

        if "experiment_name" not in self.config["OUTPUT"].keys():
            expt_name = "simple_track"
        else:
            expt_name = self.config["OUTPUT"]["experiment_name"]

        self.frame_output = FrameOutputManager(
            output_path,
            expt_name,
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
        # print(f"Hello from process {mp.current_process().name} with arg {filenames}\n")

        # Run the things
        for fnm_idx, filename in enumerate(filenames):
            frame = Frame()
            # frame.load_mwe_data(filename)
            # frame.load_data(filename)
            # frame.load_india_data(filename)
            frame.import_data_and_time(*self.loader._load(filename))
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
            # needed for tracking Features.
            # Also update the current frame with displacements for outputs
            # TODO: this choice may need to be revisited so that flow field
            # is output at the correct time (currently, it is a bit misleading)
            # that it is output at the timestep after it is used to displace features.
            prev_frame.assign_displacements(y_flow, x_flow)
            frame.assign_displacements(y_flow, x_flow)

            # Track Features between difference Frames
            # print(frame.get_time())
            self.frame_tracker.run(prev_frame, frame)

            # Output frame data to text file and field to npy
            self.frame_output.features_to_txt(frame)
            self.frame_output.fields_to_npy(frame)

            self.loading_bar.update_progress(fnm_idx + 1)
            # print("Final ids")
            # print(frame.get_features())

        self.frame_output.output_density_field(
            self.timeline, "init", centroid_only=False
        )
        self.frame_output.output_density_field(
            self.timeline, "dissipation", centroid_only=False
        )

    def run_cset(self, time_and_data_dict: dict):
        # Run the things
        for time, data in time_and_data_dict.items():
            frame = Frame()
            frame.import_data_and_time(data, time)

            # frame.load_data(filename)
            frame.identify_features(**self.config["FEATURE"])
            self.timeline.add_to_timelime(frame)

            # If this is the first frame, skip tracking
            if len(self.timeline.timeline) == 1:
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
        return self.timeline.get_timeline()

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
        if len(filenames) == 0:
            raise Exception(f"No files found in directory: {input_path}")
        return filenames

    def _read_config(self, config_path: str) -> dict:
        """
        Read config, check for necessary arguments (threshold, data paths, loader),
        return dict of parameters.

        Args:
            config_path (str):
                Path to config

        Returns:
            dict:
                Simple-Track parameters
        """
        with open(config_path, "r") as input:
            config = safe_load(input)
        self._check_config(config)
        return config

    def _check_config(self, config: dict) -> None:
        # Check required top-level sections are present
        required_sections = ["PATH", "FEATURE"]
        input_section = config.keys()
        section_check = [section in required_sections for section in input_section]
        if not all(section_check):
            raise Exception(
                f"config missing one or more required sections: {required_sections}"
            )
        # Check required parameters are present
        required_params = ["data", "loader"]
        input_keys = config["PATH"].keys()
        required_input_check = [key in input_keys for key in required_params]
        # TODO: make ConfigError in utils
        if not all(required_input_check):
            raise Exception(
                f"config missing one or more required inputs: {required_params}"
            )
        if "threshold" not in config["FEATURE"].keys():
            raise Exception("config missing required threshold input")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Running SimpleTrack requires path to at least one config")

    # For parallelisation, may need to setup the filenames here instead??

    config_paths = sys.argv[1:]
    for config_path in config_paths:
        SimpleTrack(config_path).run()
