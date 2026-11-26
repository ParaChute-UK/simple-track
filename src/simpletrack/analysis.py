import matplotlib.pyplot as plt
import numpy as np

from simpletrack.frame import Timeline


def plot_lifetime_phase_space(
    timeline: Timeline,
    feature_id: int | list = None,
    feature_aggregate: str | list = None,
    lifetime_window: int = None,
    x_property: str = "get_size",
    y_property: str = "max",
    colors: list[str] = None,
    axis: plt.Axes | None = None,
) -> list[plt.Figure, plt.Axes]:
    """
    Plot the lifetime phase space of a given feature or aggregate of features.

    Parameters
    ----------
    timeline : Timeline
        The timeline object containing the tracked features.
    feature_id : int or list, optional
        The ID(s) of the feature(s) to plot. If None, no features are plotted.
    feature_aggregate : str or list, optional
        The aggregate(s) of features to plot.
        Supported aggregates are "mean", "min", "max", "sum" or "xth percentile"
        (e.g., "90th percentile").
        If a list is provided, multiple aggregates will be plotted.
        If None, no aggregates are plotted.
    lifetime_window : int, optional
        The lifetime window to consider for the plot.
        If None, all lifetimes are considered.
    x_property : str, optional
        The property to plot on the x-axis. Default is "size".
    y_property : str, optional
        The property to plot on the y-axis. Default is "max".
    axis : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    matplotlib.figure.Figure,
        The figure containing the plot.
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = plt.gcf()

    # Get the list of features to plot and their first appearance in the timeline
    features_to_plot = []
    if feature_id is not None:
        if isinstance(feature_id, int):
            feature_id = [feature_id]
        for fid in feature_id:
            appearance_time = timeline.get_appearance_time_of_feature(fid)
            if appearance_time is None:
                print(f"Feature ID {fid} not found in the timeline.")
            else:
                features_to_plot.append(fid)

    # Build the lifetime phase space data for the specified features
    lifetime_phase_space_data = {}
    for fid in features_to_plot:
        # Add data to the lifetime phase space dictionary
        feature_data = _get_single_feature_lifetime_phase_space_data(
            timeline, fid, lifetime_window, x_property, y_property
        )
        lifetime_phase_space_data[f"Feature {fid}"] = feature_data

    # Add feature aggregates to the lifetime phase space data
    if feature_aggregate is not None:
        aggregate_data = _get_feature_aggregate_lifetime_phase_space_data(
            timeline, feature_aggregate, lifetime_window, x_property, y_property
        )
        lifetime_phase_space_data.update(aggregate_data)

    if colors is None:
        colors = [
            "#FF1F5B",
            "#00CD6C",
            "#009ADE",
            "#AF58BA",
            "#FFC61E",
            "#F28522",
            "#A0B1BA",
            "#A6761D",
        ]

    # Plot the lifetime phase space data
    for (label, data), color in zip(
        lifetime_phase_space_data.items(), colors, strict=False
    ):
        _make_phase_space_plot(
            data,
            label,
            axis,
            color,
        )

    # Add axis labels and legend
    axis.legend()
    axis.set_xlabel(x_property)
    axis.set_ylabel(y_property)

    return fig, axis


def _make_phase_space_plot(
    data_dict: dict,
    label: str,
    axis: plt.Axes,
    color: str,
):
    x_data = data_dict["x"]
    y_data = data_dict["y"]
    x_err_data = data_dict.get("xerr")
    y_err_data = data_dict.get("yerr")
    line, bars, caps = axis.errorbar(
        x_data,
        y_data,
        xerr=x_err_data,
        yerr=y_err_data,
        # ecolor=marker_colors[-1],
        marker="o",
        # mfc=marker_colors[-1],
        mec="k",
        mew=1.0,
        label=label.capitalize(),
        ls="-",
        color=color,
        # c=marker_colors[-1]
    )

    # Changes the marker of every fifth point to a square for better visibility
    fifth_indices = np.arange(4, len(x_data), 5)
    axis.plot(
        np.asarray(x_data)[fifth_indices],
        np.asarray(y_data)[fifth_indices],
        linestyle="None",
        marker="s",
        color=color,
        # markerfacecolor=marker_colors[-1],
        markeredgecolor="k",
        markeredgewidth=1.0,
        label="_nolegend_",
        zorder=10,
    )

    # Add arrows between points to indicate direction of increasing lifetime
    for i in range(len(x_data) - 1):
        axis.annotate(
            "",
            xy=(x_data[i + 1], y_data[i + 1]),
            xytext=(x_data[i], y_data[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.7),
            zorder=5,
        )

    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]


def _get_feature_aggregate_lifetime_phase_space_data(
    timeline: Timeline,
    feature_aggregate: str | list,
    lifetime_window: int = None,
    x_property: str = "get_size",
    y_property: str = "max",
) -> dict:
    """
    Get the lifetime phase space data for a given aggregate of features.

    Parameters
    ----------
    timeline : Timeline
        The timeline object containing the tracked features.
    feature_aggregate : str or list
        The aggregate(s) of features to get the lifetime phase space for.
        Supported aggregates are "mean", "min", "max", "sum" or "xth percentile"
        (e.g., "90th percentile").
        If a list is provided, multiple aggregates will be returned.
    lifetime_window : int, optional
        The lifetime window to consider for the data. If None, all lifetimes are
        considered.
    x_property : str, optional
        The property to use for the x-axis. Default is "size".
    y_property : str, optional
        The property to use for the y-axis. Default is "max".

    Returns
    -------
    dict
        A dictionary containing the x and y values for the specified feature's
        lifetime phase space.
        Example: {"x": [...], "y": [...]}
    """
    aggregate_map = {
        "mean": np.nanmean,
        "min": np.nanmin,
        "max": np.nanmax,
        "sum": np.nansum,
        "percentile": None,  # defined below,
    }

    if isinstance(feature_aggregate, str):
        feature_aggregate = [feature_aggregate]

    if lifetime_window is None:
        lifetime_window = -1  # Use all lifetimes if no window is specified

    aggregate_data = {}
    for aggregate in feature_aggregate:
        # Construct percentile aggregate function
        if "percentile" in aggregate or "Percentile" in aggregate:
            percentile_value = "".join(x for x in aggregate if x.isdigit())
            percentile_value = int(percentile_value) if percentile_value else 50

            def aggregate_func(x, percentile_value=percentile_value, axis=1):
                return np.nanpercentile(x, percentile_value, axis=axis)
        else:
            aggregate_func = aggregate_map.get(aggregate)

        if aggregate_func is None:
            raise ValueError(
                f"Unsupported feature aggregate: {aggregate}. "
                f"Supported aggregates are: {list(aggregate_map.keys())}."
            )

        # Get the lifetime phase space data for the specified aggregate
        feature_data = _get_feature_data_by_lifetime(
            timeline, property=[x_property, y_property]
        )
        x_values = feature_data[x_property]
        y_values = feature_data[y_property]

        # Unpack these dictionaries into lists of values for aggregation
        # Also apply lifetime window if specified
        x_values = [sublist[:lifetime_window] for sublist in x_values.values()]
        y_values = [sublist[:lifetime_window] for sublist in y_values.values()]

        # Pad list to ensure all sublists are the same length for aggregation
        if lifetime_window > 0:
            max_length = max(len(sublist) for sublist in x_values)
            x_values = [
                np.pad(
                    np.array(sublist, dtype=float),
                    pad_width=(0, max_length - len(sublist)),
                    mode="constant",
                    constant_values=np.nan,
                )
                for sublist in x_values
            ]
            y_values = [
                np.pad(
                    np.array(sublist, dtype=float),
                    pad_width=(0, max_length - len(sublist)),
                    mode="constant",
                    constant_values=np.nan,
                )
                for sublist in y_values
            ]

        # Apply the aggregate function to the x and y values
        aggregated_x = aggregate_func(x_values, axis=0)
        aggregated_y = aggregate_func(y_values, axis=0)

        # Store the aggregated data in the dictionary
        aggregate_data[aggregate] = {"x": aggregated_x, "y": aggregated_y}

    return aggregate_data


def _get_feature_data_by_lifetime(
    timeline: Timeline,
    property: str | list = "get_size",
) -> dict:
    """
    Get the requested feature data compiled by lifetime for all features
    in the timeline. Returns a dict where each key is a feature ID
    and the values are lists of the requested property.

    If multiple properties are requested, a nested dict is returned
    where the first level keys are property names and the second level keys
    are feature IDs.
    """
    if isinstance(property, str):
        property = [property]

    # Construct nested dict of property -> feature_id (-> list of values later)
    features_data = {prop: {} for prop in property}

    for frame in timeline.get_timeline().values():
        for feature in frame.features.values():
            fid = feature.id

            for prop in property:
                if fid not in features_data[prop]:
                    features_data[prop][fid] = []

            # Get the requested property value
            for prop in property:
                prop_value = getattr(feature, prop)
                if callable(prop_value):
                    prop_value = prop_value()

                features_data[prop][fid].append(prop_value)

    return features_data


def _get_single_feature_lifetime_phase_space_data(
    timeline: Timeline,
    feature_id: int,
    lifetime_window: int = None,
    x_property: str = "get_size",
    y_property: str = "max",
) -> dict:
    """
    Get the lifetime phase space data for a single feature.

    Parameters
    ----------
    timeline : Timeline
        The timeline object containing the tracked features.
    feature_id : int
        The ID of the feature to get the lifetime phase space for.
    lifetime_window : int, optional
        The lifetime window to consider for the data. If None, all lifetimes are
        considered.
    x_property : str, optional
        The property to use for the x-axis. Default is "size".
    y_property : str, optional
        The property to use for the y-axis. Default is "max".

    Returns
    -------
    dict
        A dictionary containing the x and y values for the specified feature's
        lifetime phase space.
        Example: {"x": [...], "y": [...]}
    """
    frame_times = list(timeline.get_timeline().keys())
    appearance_time = timeline.get_appearance_time_of_feature(feature_id)

    feature_data = {"x": [], "y": []}

    # Get the times to iterate over based on lifetime_window
    start_idx = frame_times.index(appearance_time)
    if lifetime_window is not None:
        end_idx = min(start_idx + lifetime_window, len(frame_times))
    else:
        end_idx = len(frame_times)

    # Iterate over the relevant frame times and collect the x and y properties
    for time in frame_times[start_idx:end_idx]:
        frame = timeline.get_frame(time)
        feature = frame.get_feature(feature_id)
        if feature is not None:
            x_value = getattr(feature, x_property)
            y_value = getattr(feature, y_property)
            if callable(x_value):
                x_value = x_value()
            if callable(y_value):
                y_value = y_value()
            feature_data["x"].append(x_value)
            feature_data["y"].append(y_value)

    return feature_data
