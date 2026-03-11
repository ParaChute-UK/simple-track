from frame import Frame, Timeline
from pathlib import Path
import numpy as np
from utils import check_arrays
from typing import Union
import csv


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

            frame_features_dict = frame.get_features()

            for feature_id in sorted(frame_features_dict):
                feature = frame_features_dict[feature_id]
                output_line = feature.summarise(output_type="str")
                output_file.write(output_line + "\n")

    def features_to_csv(self, frame: Frame) -> None:
        frame_time = frame.get_time()
        frame_time_str = frame_time.strftime("%Y%m%d_%H%M")
        output_fnm = f"{self.output_path}/frame_{frame_time_str}.csv"

        with open(output_fnm, "w") as output_file:
            # Write headers
            writer = csv.writer(output_file)
            writer.writerow([self.expt_name])
            if self.config_path is not None:
                writer.writerow([f"Config path: {self.config_path}"])
            if self.start_time is not None:
                writer.writerow([f"Start time: {self.start_time}"])
            writer.writerow([f"Frame time: {frame_time_str}"])
            writer.writerow([f"Total tracked features: {frame.get_max_id()}"])

            # Write data
            frame_features_dict = frame.get_features()
            # Get data headers by looking at any feature in the frame features dict
            random_feature = frame_features_dict[list(frame_features_dict.keys())[0]]
            data_headers = random_feature.summarise("dict").keys()
            dict_writer = csv.DictWriter(output_file, fieldnames=data_headers)
            dict_writer.writeheader()

            for feature_id in sorted(frame_features_dict):
                feature = frame_features_dict[feature_id]
                dict_writer.writerow(feature.summarise("dict"))

    def fields_to_npy(self, frame: Frame) -> None:
        """
        Output feature and lifetime fields to .npy files

        Args:
            frame (Frame): _description_
        """
        outputs = {
            "features": frame.get_feature_field(),
            "lifetime": frame.get_lifetime_field(),
            "y_flow": frame.get_flow()[0],
            "x_flow": frame.get_flow()[1],
        }
        frame_time = frame.get_time()
        frame_time_str = frame_time.strftime("%Y%m%d_%H%M")
        for output_fnm, output in outputs.items():
            full_fnm = f"{self.output_path}/{output_fnm}_{frame_time_str}.npy"
            np.save(full_fnm, output)

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


class LoadOutput:
    """
    Contains functionality for reading previous outputs back into a Timeline object (contanining
    Frames of field and Feature data) for further inspection and analysis.
    """

    def __init__(self, data_path: Union[str | Path]):
        self.path = Path(data_path)

    def load(self) -> Timeline:
        timeline = Timeline()

        all_output_fields = self.path.rglob("*.npy")
        all_txt_files = self.path.rglob("*.txt")
