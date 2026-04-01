import csv
import datetime
from ast import literal_eval
from pathlib import Path
from typing import Union

import numpy as np

from simpletrack.feature import Feature
from simpletrack.frame import Frame, Timeline
from simpletrack.utils import check_arrays


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
        self.strftime = "%Y%m%d_%H%M"

    def features_to_txt(self, frame: Frame) -> None:
        """
        Outputs contents of input Frame to a text file

        Args:
            frame (Frame): _description_
        """
        frame_time = frame.time
        frame_time_str = frame_time.strftime(self.strftime)
        output_fnm = f"{self.output_path}/frame_{frame_time_str}.txt"

        with open(output_fnm, "w") as output_file:
            output_file.write(self.expt_name + "\n")
            if self.config_path is not None:
                output_file.write(f"Config path: {self.config_path}\n")
            if self.start_time is not None:
                output_file.write(f"Start time: {self.start_time}\n")
            output_file.write(f"Frame time: {frame_time_str}\n")
            output_file.write(f"Total tracked features: {frame.max_id}\n")

            frame_features_dict = frame.features

            for feature_id in sorted(frame_features_dict):
                feature = frame_features_dict[feature_id]
                output_line = feature.summarise(output_type="str")
                output_file.write(output_line + "\n")

    def features_to_csv(self, frame: Frame) -> None:
        frame_time = frame.time
        frame_time_str = frame_time.strftime(self.strftime)
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
            writer.writerow([f"Total tracked features: {frame.max_id}"])

            # Write data
            frame_features_dict = frame.features
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
            "features": [frame.feature_field, "%.6e"],
            "lifetime": [frame.lifetime_field, "%.4e"],
            "y-flow": [frame.get_flow()[0], "%.2e"],
            "x-flow": [frame.get_flow()[1], "%.2e"],
        }
        frame_time = frame.time
        frame_time_str = frame_time.strftime("%Y%m%d_%H%M")
        for output_fnm, [output, output_fmt] in outputs.items():
            if output is None:
                continue
            full_fnm = f"{self.output_path}/{output_fnm}_{frame_time_str}.field"
            np.savetxt(full_fnm, output, fmt=output_fmt)

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
            *[frame.feature_field for frame in all_frames],
            ndim=2,
            equal_shape=True,
            non_negative=True,
        )

        # If above check passes, can make storage array from first frame field
        field_shape = (len(all_frames), *all_frames[0].feature_field.shape)
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

    def __init__(self, st_data_path: Union[str | Path]):
        self.path = Path(st_data_path)
        self.strftime = "%Y%m%d_%H%M"
        # Links field type names in outputs to attribute names in Frame
        self.field_attributes = {
            "features": "feature_field",
            "lifetime": "lifetime_field",
            "x-flow": "x_flow",
            "y-flow": "y_flow",
        }

    def load_to_timeline(self) -> Timeline:
        timeline = Timeline()

        # Get list of times from output fields.
        frame_times = self.get_frame_times_from_field_filenames()

        # Load blank Frames into Timeline
        for frame_time in frame_times:
            frame = Frame()
            frame.set_time(frame_time)
            timeline.add_to_timelime(frame)

        # Load fields into blank Frames
        self.load_frame_fields(timeline)
        # Load raw data into frame
        # TODO: will need config file for this to determine
        # if a Loader has been used, and location of input data
        # self.load_raw_fields(timeline)

        # Populate features in each Frame
        for frame in timeline.get_timeline().values():
            frame.populate_features()

        # Finally, fill these Features with data loaded from outputs
        self.load_feature_data(timeline)
        return timeline

    def load_feature_data(self, timeline: Timeline) -> None:
        # Get list of headers from blank Feature class
        blank_feature = Feature(
            id=1,
            feature_coords=np.array(((0, 0), (0, 0))),
            time=datetime.datetime.now(),
        )
        headers = blank_feature.summarise(headers_only=True)
        # Set the number of headers to skip in each csv file
        number_header_rows = 5

        for frame_time, frame in timeline.get_timeline().items():
            # Load all data for the current time
            frame_time_str = frame_time.strftime(self.strftime)
            frame_time_fnames = self.path.rglob(f"*{frame_time_str}.csv")

            for fname in frame_time_fnames:
                # Read data from output
                with open(fname, "r") as csv_file:
                    reader = csv.DictReader(csv_file, fieldnames=headers)
                    all_feature_data = [
                        row
                        for row_idx, row in enumerate(reader)
                        if row_idx > number_header_rows
                    ]

                # Add data to feature object in Frame
                for feature_data in all_feature_data:
                    id = int(feature_data["id"])
                    feature = frame.get_feature(id)

                    # Loop over all features, set attribute to value
                    for property, value in feature_data.items():
                        if property == "id" or property == "centroid":
                            continue

                        if len(value) == 0:
                            setattr(feature, property, None)
                        else:
                            # Literal eval converts str to inferred python type
                            setattr(feature, property, literal_eval(value))

    def load_frame_fields(self, timeline: Timeline) -> None:
        for frame_time, frame in timeline.get_timeline().items():
            # Load all data for the current time
            frame_time_str = frame_time.strftime(self.strftime)
            frame_time_fnames = self.path.rglob(f"*{frame_time_str}.field")

            for fname in frame_time_fnames:
                ftype = fname.name.split("_")[0]
                # Set the relevant attribute of frame, as mapped using
                # self.field_attributes
                setattr(frame, self.field_attributes[ftype], np.loadtxt(fname))

    def get_frame_times_from_field_filenames(self) -> list:
        """
        Using a list of all field filenames from a given run of SimpleTrack,
        determine the frame times. Checks whether data is present for all
        field types

        Returns:
            list: list of all frame times as datetime.datetime object
        """
        # Containing dict for these times
        times_from_each_field_type = {}

        # Iterate over each field type to find time the data is available for
        for ftype in self.field_attributes.keys():
            # Setup containing array for times
            field_filenames = self.path.rglob(f"{ftype}*.field")
            field_times = []

            for field_fname in sorted(field_filenames):
                fname_parts = str(field_fname.name).split("_")
                yyyymmdd = fname_parts[1]
                hhmm = fname_parts[2]
                field_times.append(
                    datetime.datetime(
                        year=int(yyyymmdd[0:4]),
                        month=int(yyyymmdd[4:6]),
                        day=int(yyyymmdd[6:8]),
                        hour=int(hhmm[0:2]),
                        minute=int(hhmm[2:4]),
                    )
                )
            times_from_each_field_type[ftype] = field_times

        # Check times from each field type to check consistency
        # If all times are the same (have previously been sorted), should only
        # be one set of unqiue dict values
        times_list = times_from_each_field_type.values()
        times_set = [set(tuple(times)) for times in times_list]
        if len(times_set) == 1:
            # All times are the same, can return any list
            return times_from_each_field_type["features"]

        else:
            # Get the longest list to return
            max_arr_key = max(
                times_from_each_field_type,
                key=lambda x: len(times_from_each_field_type[x]),
            )
            return times_from_each_field_type[max_arr_key]
