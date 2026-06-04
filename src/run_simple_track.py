import argparse

from simpletrack import Tracker


def run_tracking():
    # TODO: make argparser default way of handling inputs, including configs and loaders
    # Need to make sure that changes don't affect pyproject.toml entry points
    # easiest just to pass the parser in to run_tracking.
    parser = argparse.ArgumentParser(description="Run Simple-Track")
    parser.add_argument(
        "configs", nargs="+", help="Path to one or more yaml config files."
    )
    parser.add_argument(
        "-i",
        "--path",
        required=False,
        help="Path to input data. If not supplied will use path in config file",
    )
    parser.add_argument(
        "-l",
        "--loader",
        required=False,
        help="Path to loader function in format 'path|func_name'",
    )
    parser.add_argument(
        "-ia",
        "--iterate_over_array",
        required=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, will iterate over a single array rather than multiple files.",
    )
    parser.add_argument(
        "-dim",
        "--iterating_dim",
        required=False,
        type=int,
        help="If --iterate_over_arrays flagged, sets the dimension to iterate over",
    )
    args = parser.parse_args()

    # If --iterate-over-arrays flagged and no --iterating-dim supplied, raise error
    if args.iterate_over_array and args.iterating_dim is None:
        parser.error("--iterate_over_array requires --iterating_dim")

    # All optional CLI args will be passed in as kwargs to Tracker config
    kwarg_names = ["path", "loader", "iterate_over_array", "iterating_dim"]
    kwarg_vals = [getattr(args, kw_name) for kw_name in kwarg_names]
    kwargs = {
        kw_name: kw_val
        for kw_name, kw_val in zip(kwarg_names, kwarg_vals, strict=True)
        if kw_val is not None
    }

    for config_path in args.configs:
        # With None passed into run method, uses input path in config
        Tracker(config_path, **kwargs).run()


if __name__ == "__main__":
    run_tracking()
