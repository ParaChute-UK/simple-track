import sys

from simpletrack.track import Tracker

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Running SimpleTrack requires path to at least one config")

    config_paths = sys.argv[1:]
    for config_path in config_paths:
        # With None passed into run method, uses input path in config
        Tracker(config_path).run()
