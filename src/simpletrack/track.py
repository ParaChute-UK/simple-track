"""
Run the SimpleTrack algorithm to track objects through a sequence of images
"""

from pathlib import Path
from typing import Union

from yaml import safe_load

from simpletrack.flow_solver import FlowSolver
from simpletrack.frame import Frame, Timeline
from simpletrack.frame_output import FrameOutputManager
from simpletrack.frame_tracker import FrameTracker
from simpletrack.load import ConfigError, DictIterator, LoadingBar, get_loader


class Tracker:
    def __init__(self, config_input: Union[str | dict]) -> None:
        """
        Initialize SimpleTrack with configuration file

        Args:
            config_iput (str|dict):
                If str, provides Path to the configuration file
                If dict, containts pre-loaded config parameters
        """
        if isinstance(config_input, str):
            config_path = config_input
            self.config = self._read_config(config_input)
        elif isinstance(config_input, dict):
            config_path = None
            self._check_config(config_input)
            self.config = config_input
        else:
            raise TypeError(
                f"Expected config_input type str or dict, got {type(config_input)}"
            )

        self.start_time = None  # Will be set during run()
        self.timeline = Timeline()

        self.file_type = None
        if "INPUT" in self.config.keys():
            if "file_type" in self.config["INPUT"].keys():
                self.file_type = self.config["INPUT"]["file_type"]

        if "FLOW_SOLVER" in self.config.keys():
            self.flow_solver = FlowSolver(**self.config["FLOW_SOLVER"])
        else:
            self.flow_solver = FlowSolver()

        # Whether to skip tracking and just output Feature properties
        self.skip_tracking = False
        if "TRACKING" in self.config.keys():
            self.frame_tracker = FrameTracker(**self.config["TRACKING"])
            if "skip_tracking" in self.config["TRACKING"].keys():
                self.skip_tracking = self.config["TRACKING"]["skip_tracking"]
        else:
            self.frame_tracker = FrameTracker()

        if "OUTPUT" in self.config.keys():
            if "path" not in self.config["OUTPUT"].keys():
                output_path = "./output"
            else:
                output_path = self.config["OUTPUT"]["path"]

        if "OUTPUT" in self.config.keys():
            if "experiment_name" in self.config["OUTPUT"].keys():
                expt_name = self.config["OUTPUT"]["experiment_name"]
        else:
            expt_name = "Simple-Track Experiment"

        # Output only if flagged in config
        self.frame_output = None
        if "OUTPUT" in self.config.keys():
            if self.config["OUTPUT"]["save_data"]:
                self.frame_output = FrameOutputManager(
                    output_path,
                    expt_name,
                    self.start_time,
                    config_path,
                )

    def run(self, input_data: Union[list[str] | dict] = None):
        """
        Runs SimpleTrack using the designated config options.

        Input data can either be read in from filenames (list(str)) or provided
        as input using dictionary

        If input_data is None, SimpleTrack finds all valid files in ["PATH"]["data]
        config input using "SimpleTrack.get_filenames_from_input_path"

        If data is being read in using filenames, there must also be an associated
        Loader class argument in config["PATH"]["loader"] that defines how the data
        should be pre-processed and how the validity time should be determined.
        Filenames should be ordered by time. Loaded data will be checked for consistent
        array shapes. See docs or src.load.py for more.

        If data is being provided as input using dict, it should be passed
        with the respective datetime object as the key, and the numpy array to run
        tracking on as the value. This will not use a predetermined Loader class to
        load the data, although the same checks on consistent array shapes will be applied.
        """
        if input_data is None:
            input_data = self.get_filenames_from_input_path(file_type=self.file_type)

        if isinstance(input_data, list):
            valid_types = (str, Path)
            if not all([isinstance(fnm, valid_types) for fnm in input_data]):
                types = [type(fnm) for fnm in input_data]
                raise TypeError(
                    f"If input_data is passed a list, it must only contain str, got {types}"
                )
            self.loading_bar = LoadingBar(total=len(input_data))
            self.loader = get_loader(self.config["INPUT"]["loader"])(input_data)

        elif isinstance(input_data, dict):
            self.loading_bar = LoadingBar(total=len(input_data.values()))
            self.loader = DictIterator(input_data)

        else:
            raise TypeError(
                f"Expected input_data type list(str) or dict, got {type(input_data)}"
            )
        # print(f"Hello from process {mp.current_process().name} with arg {filenames}\n")

        for fnm_idx, time_and_data in enumerate(self.loader):
            if self.start_time is None:
                self.start_time = time_and_data[0]

            frame = Frame()
            frame.import_time_and_data(*time_and_data)
            frame.identify_features(**self.config["FEATURE"])
            self.timeline.add_to_timelime(frame)

            # If this is the first frame, skip tracking
            if len(self.timeline.timeline) == 1 or self.skip_tracking:
                self.loading_bar.update_progress(fnm_idx + 1)
                # Output frame data to text file or npy file if flagged
                if self.frame_output is not None:
                    self.frame_output.features_to_txt(frame)
                    self.frame_output.features_to_csv(frame)
                    self.frame_output.fields_to_npy(frame)
                continue

            # Now run flow solver between previous and current frame
            prev_frame = self.timeline.get_previous_frame(frame.get_time())
            y_flow, x_flow = self.flow_solver.analyse_flow(prev_frame, frame)

            # Update the previous Frame with these displacements which is
            # needed for tracking Features.
            if y_flow is not None or x_flow is not None:
                prev_frame.assign_displacements(y_flow, x_flow)
                frame.assign_displacements(y_flow, x_flow)

            # Set max id for assigning to new features
            frame.set_max_id(prev_frame.get_max_id())
            # Track Features between difference Frames
            self.frame_tracker.run(prev_frame, frame)

            # Output frame data to text file and field to npy if flagged
            if self.frame_output is not None:
                self.frame_output.features_to_txt(frame)
                self.frame_output.features_to_csv(frame)
                self.frame_output.fields_to_npy(frame)

            self.loading_bar.update_progress(fnm_idx + 1)

        # Output additional fields if flagged
        if self.frame_output is not None:
            self.frame_output.output_density_field(
                self.timeline, "init", centroid_only=False
            )
            self.frame_output.output_density_field(
                self.timeline, "dissipation", centroid_only=False
            )
        return self.timeline

    # def run_parallel(self, processes=4):
    #     # Split filenames into chunks for each process
    #     chunk_size = len(self.filenames) // processes
    #     filename_chunks = [
    #         self.filenames[i : i + chunk_size]
    #         for i in range(0, len(self.filenames), chunk_size)
    #     ]

    #     with mp.Pool(processes=processes) as pool:
    #         # TODO: figure out how to do this with the new version of run above, where
    #         # not having filename inputs means it tries to get it from config...
    #         pool.map(self.run, filename_chunks)

    #     # TODO: then need a way to make the results consistent between
    #     # different chunks.
    #     # I.e., if the last event of chunk 1 contains a storm that is
    #     # also present in the first event of chunk 2, then the chunk 2
    #     # storm needs to have a consistent ID, needs to have updated lifetimes
    #     # etc.
    #     # This is apparently already solved in Will Keats/Callum Scullion MO
    #     # code so don't need to reinvent the wheel here.

    def get_filenames_from_input_path(
        self, input_path: str = None, file_type: str = None
    ) -> list:
        """
        Get a list of filenames from a given input path matching a given
        file type

        Args:
            input_path (str, optional):
                Input path to search for filenames
                Defaults to self.config["INPUT"]["path"]
            file_type (str, optional):
                File type to search input_path for
                Defaults to .nc

        Raises:
            Exception: _description_

        Returns:
            list: _description_
        """
        if input_path is None:
            input_path = self.config["INPUT"]["path"]

        supported_filetypes = [".nc"]
        if file_type is not None:
            if isinstance(file_type, str):
                supported_filetypes.append(file_type)
            elif isinstance(file_type, list):
                if not all([isinstance(val, str) for val in file_type]):
                    raise TypeError(
                        f"Expected list to contain only str, gt {[type(val) for val in file_type]}"
                    )
                for ftype in file_type:
                    supported_filetypes.append(ftype)
            else:
                raise TypeError(f"Expected list or str, got {type(file_type)}")

        filenames = sorted(
            [
                p
                for p in Path(input_path).iterdir()
                if p.is_file() and p.suffix in supported_filetypes
            ]
        )
        if len(filenames) == 0:
            raise FileNotFoundError(f"No files found in directory: {input_path}")
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
        required_sections = ["FEATURE"]
        input_section = config.keys()
        section_check = [section in input_section for section in required_sections]
        if not all(section_check):
            raise ConfigError(
                f"config missing one or more required sections: {required_sections}"
            )
        # # Check required parameters are present
        # required_params = ["data"]
        # input_keys = config["PATH"].keys()
        # required_input_check = [key in input_keys for key in required_params]

        # if not all(required_input_check):
        #     raise ConfigError(
        #         f"config missing one or more required inputs: {required_params}"
        #     )
        if "threshold" not in config["FEATURE"].keys():
            raise ConfigError("config missing required threshold input")
