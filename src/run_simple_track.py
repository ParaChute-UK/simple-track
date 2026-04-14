import argparse

from simpletrack import Tracker


def run_tracking():
    # TODO: make argparser default way of handling inputs, including configs and loaders
    # Need to make sure that changes don't affect pyproject.toml entry points
    # easiest just to pass the parser in to run_tracking.
    parser = argparse.ArgumentParser(description="Run Simple-Track")
    parser.add_argument("configs", help="Path to one or more yaml config files.")
    args = parser.parse_args()

    for config_path in args.configs:
        # With None passed into run method, uses input path in config
        Tracker(config_path).run()


if __name__ == "__main__":
    run_tracking()
