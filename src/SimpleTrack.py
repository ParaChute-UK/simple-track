"""
Run the SimpleTrack algorithm to track objects through a sequence of images
"""

import sys
from yaml import safe_load
from pathlib import Path

from Event import EventTimeline, Event


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
        self.filenames = self.__get_files_from_input_path(self.config["PATH"]["data"])

        self.events = EventTimeline()

    def run(self):
        # Run the things
        for filename in self.filenames:
            # Load the data
            event = Event()
            event.load_data(filename)
            event.identify_features(self.config["FEATURE"])
            self.events.add_to_timelime(event)

    def __get_files_from_input_path(self, input_path: str) -> list:
        supported_filetypes = [".nc"]
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

    config_path = sys.argv[1]
    SimpleTrack(config_path).run()
