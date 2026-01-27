from Frame import Frame
from pathlib import Path
import numpy as np


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
