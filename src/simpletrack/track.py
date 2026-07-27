"""
Run the SimpleTrack algorithm to track objects through a sequence of images
"""

from glob import glob
from pathlib import Path

from simpletrack.flow_solver import FlowSolver
from simpletrack.frame import Frame, Timeline
from simpletrack.frame_output import FrameOutputManager
from simpletrack.frame_tracker import FrameTracker
from simpletrack.load import (
    ArrayIterator,
    ConfigError,
    DictIterator,
    FilenameIterator,
    LoadingBar,
)


class Tracker:
    """
    Simple-Track manager controlling inputs, processing, outputs
    """

    def __init__(self, config_input: str | dict, **kwargs) -> None:
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

        # Get "max_frames" from config if present, otherwise default to None (no limit)
        if "TIMELINE" in self.config:
            max_frames = self.config["TIMELINE"].get("max_frames", None)
            self.timeline = Timeline(max_frames=max_frames)
        else:
            self.timeline = Timeline()

        if "INPUT" in self.config:
            self.loader = self.config["INPUT"].get("loader", None)
            self.iterate_over_array = self.config["INPUT"].get(
                "iterate_over_array", False
            )
            self.iterating_dim = self.config["INPUT"].get("iterating_dim", None)
            self.path = self.config["INPUT"].get("path", None)

        else:
            self.loader = None
            self.iterate_over_array = False
            self.iterating_dim = None
            self.path = None

        # Override any INPUT attributes with values from CLI (kwargs)
        for attr_name, attr_val in kwargs.items():
            setattr(self, attr_name, attr_val)

        if "FLOW_SOLVER" in self.config:
            # Allows empty config to be passed into FlowSolver,
            # which will then use default values
            flow_config = self.config["FLOW_SOLVER"] or {}
            self.flow_solver = FlowSolver(**flow_config)
        else:
            self.flow_solver = FlowSolver()

        if "TRACKING" in self.config:
            self.frame_tracker = FrameTracker(**self.config["TRACKING"])
        else:
            self.frame_tracker = FrameTracker()

        if "OUTPUT" in self.config:
            self.skip_tracking = self.config["OUTPUT"].get("skip_tracking", False)
            output_path = self.config["OUTPUT"].get("path", "./output")
            expt_name = self.config["OUTPUT"].get(
                "experiment_name", "Simple-Track Experiment"
            )
        else:
            self.skip_tracking = False
            output_path = "./output"
            expt_name = "Simple-Track Experiment"

        # Output only if flagged in config
        self.frame_output = None
        if "OUTPUT" in self.config:
            if self.config["OUTPUT"]["save_data"]:
                save_raw_data = self.config["OUTPUT"].get("output_raw_data", True)
                if not save_raw_data:
                    msg = (
                        "Warning: disabling output of raw data will prevent "
                        "re-loading of this data for further analysis."
                    )
                    print(msg)

                self.frame_output = FrameOutputManager(
                    output_path,
                    expt_name,
                    self.start_time,
                    config_path,
                    output_raw_data=save_raw_data,
                )

    def run(self, input_data: list[str] | dict = None) -> Timeline:
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
        load the data, although the same checks on consistent array shapes
        will be applied.

        Returns Timeline object containing Frames of data and tracked Features.
        """
        # Get input files to load if inputs not provided
        if input_data is None:
            input_data = self.get_filenames_from_input_path(self.path)

        self._setup_loaders(input_data)

        # print(f"Hello from proc {mp.current_process().name} with arg {filenames}\n")

        # Iterate through sorted input data, perform tracking, output results if flagged
        for fnm_idx, time_and_data in enumerate(self.loader):
            if self.start_time is None:
                self.start_time = time_and_data[0]

            # Import data to Frame and add to Timeline
            frame = Frame()
            frame.import_time_and_data(*time_and_data)
            frame.identify_features(**self.config["FEATURE"])
            self.timeline.add_to_timelime(frame)

            # If this is the first frame or tracking is disabled, skip tracking
            if len(self.timeline.timeline) == 1 or self.skip_tracking:
                self.loading_bar.update_progress(fnm_idx + 1)
                # Output frame data to text file or npy file if flagged
                if self.frame_output is not None:
                    self.frame_output.features_to_txt(frame)
                    self.frame_output.features_to_csv(frame)
                    self.frame_output.fields_to_npy(frame)
                continue

            # Now run flow solver between previous and current frame
            prev_frame = self.timeline.get_previous_frame(frame.time)
            # Set max id for assigning to new features
            if prev_frame.max_id is not None:
                frame.max_id = prev_frame.max_id
            # Get the flow field that translates features between the two frames
            y_flow, x_flow = self.flow_solver.analyse_flow(prev_frame, frame)

            # Update the current Frame with these displacements
            if y_flow is not None or x_flow is not None:
                frame.assign_displacements(y_flow, x_flow)

            # Match Features between Frames
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

    def get_filenames_from_input_path(self, input_path: str = None) -> list:
        """
        Get a list of filenames from a given input path matching a given
        file type

        Args:
            input_path (str, optional):
                Input path to search for filenames
                Defaults to self.config["INPUT"]["path"]
        """
        if input_path is None:
            input_path = self.config["INPUT"].get("path", None)

        if input_path is None:
            raise ConfigError("'INPUT''path' required in input config but not found")

        filenames = sorted([Path(path) for path in glob(input_path)])
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

        from yaml import safe_load

        with open(config_path) as input:
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
        if "threshold" not in config["FEATURE"]:
            raise ConfigError("config missing required threshold input")

    def _setup_loaders(self, input_data) -> None:
        # Check type of input data and set up loader accordingly
        if isinstance(input_data, list):
            valid_types = (str, Path)
            if not all([isinstance(fnm, valid_types) for fnm in input_data]):
                types = [type(fnm) for fnm in input_data]
                raise TypeError(
                    f"If input_data is list it must only contain str, got {types}"
                )
            self.loading_bar = LoadingBar(total=len(input_data))
            # Check type of loader to use

            if self.loader is None:
                raise ValueError(
                    "loader is required to load input data. See docs for more"
                )

            # Setup ArrayIterator
            if self.iterate_over_array:
                if self.iterating_dim is None:
                    raise ValueError("Iterating over arrays requires iterator_dim")
                if not isinstance(self.iterating_dim, int):
                    raise TypeError(
                        f"iterator_dim must be type int, got {type(self.iterating_dim)}"
                    )
                self.loader = ArrayIterator(input_data, self.loader, self.iterating_dim)
            # Setup FilenameIterator
            else:
                self.loader = FilenameIterator(input_data, self.loader)

        elif isinstance(input_data, dict):
            self.loading_bar = LoadingBar(total=len(input_data.values()))
            self.loader = DictIterator(input_data)

        else:
            raise TypeError(
                f"Expected input_data type list(str) or dict, got {type(input_data)}"
            )
