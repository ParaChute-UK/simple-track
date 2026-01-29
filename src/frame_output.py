from frame import Frame, Timeline
from pathlib import Path
import numpy as np
from utils import check_arrays
import matplotlib


class FrameOutputManager:
    def __init__(
        self,
        output_path: str = "./output",
        expt_name: str = "default",
        start_time: str = None,
        config_path: str = None,
    ):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.expt_name = expt_name
        self.start_time = start_time
        self.config_path = config_path

    def features_to_txt(self, frame: Frame) -> None:
        """
        Outputs contents of input Frame to a text file

        Args:
            frame (Frame): _description_
        """
        # Check output path exists

        frame_time = frame.get_time()
        frame_time_str = frame_time.strftime("%Y%m%d_%H%M")
        output_fnm = f"{self.output_path}/frame_{frame_time_str}.txt"
        with open(output_fnm, "w") as output_file:
            output_file.write(self.expt_name + "\n")
            if self.config_path is not None:
                output_file.write(f"Config path: {self.config_path}\n")
            if self.start_time is not None:
                output_file.write(f"Start time: {self.start_time}\n")
            output_file.write(f"Frame time: {frame_time_str}\n")
            output_file.write(f"Total tracked features: {frame.get_max_id()}\n")

            for feature in frame.get_features().values():
                output_line = feature.summarise(output_type="str")
                output_file.write(output_line + "\n")

    def fields_to_npy(self, frame: Frame) -> None:
        """
        Output feature and lifetime fields to .npy files

        Args:
            frame (Frame): _description_
        """
        frame_time = frame.get_time()
        frame_time_str = frame_time.strftime("%Y%m%d_%H%M")
        feature_output_fnm = f"{self.output_path}/features_{frame_time_str}.npy"
        lifetime_output_fnm = f"{self.output_path}/lifetimes_{frame_time_str}.npy"
        np.save(feature_output_fnm, frame.get_feature_field())
        np.save(lifetime_output_fnm, frame.get_lifetime_field())

    def output_density_field(
        self, timeline: Timeline, field_type: str, centroid_only: bool = True
    ):
        """
        Loops over all Frames in Timeline, makes density plot of areas
        where new Features are being created

        Args:
            timeline (Timeline): _description_
        """
        valid_types = ["init", "dissipation"]
        if field_type not in valid_types:
            raise ValueError(
                f"field_type ({field_type}) not in valid_types {valid_types}"
            )

        all_frames = list(timeline.get_timeline().values())
        if not all((isinstance(frame, Frame) for frame in all_frames)):
            return TypeError(f"Expected all Frames, got {all_frames}")

        check_arrays(
            *[frame.get_feature_field() for frame in all_frames],
            ndim=2,
            equal_shape=True,
            non_negative=True,
        )

        # If above check passes, can make storage array from first frame field
        field_shape = (len(all_frames), *all_frames[0].get_feature_field().shape)
        field_density = np.zeros(field_shape)

        for frame_idx, frame in enumerate(all_frames):
            field = frame.get_field(field_type, centroid_only=centroid_only)
            field_density[frame_idx, ...] = field

        output_fnm = f"{self.output_path}/{field_type}_density.npy"
        np.save(output_fnm, field_density)
