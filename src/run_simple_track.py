import argparse
import sys

from simpletrack import Tracker


def run_tracking():
    if len(sys.argv) < 2:
        raise Exception("Running SimpleTrack requires path to at least one config")

    config_paths = sys.argv[1:]
    for config_path in config_paths:
        # With None passed into run method, uses input path in config
        Tracker(config_path).run()


if __name__ == "__main__":
    # TODO: make argparser default way of handling inputs, including configs and loaders
    # Need to make sure that changes don't affect pyproject.toml entry points
    # easiest just to pass the parser in to run_tracking.
    msg = "Run Simple-Track. Requires path to at least one yaml config file"
    parser = argparse.ArgumentParser(description=msg)
    run_tracking()
