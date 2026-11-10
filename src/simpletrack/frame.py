import datetime as dt

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from numpy.typing import NDArray

from simpletrack.exceptions import FeaturesNotFoundError
from simpletrack.feature import Feature
from simpletrack.utils import check_arrays, check_valid_ids, native


class Frame:
    """
    Object for storing data and methods related to a single timestep of data. This
    includes the raw data, the feature field, the feature lifetime field, and a dict
    of Feature objects for each feature identified in the frame. The Frame class also
    includes methods for identifying features in the frame, and for assigning
    pre-calculated motion vectors to each Feature.
    """

    def __init__(self):
        self._time = None
        self._raw_field = None
        self._feature_field = None
        self._lifetime_field = None
        self._max_id = None
        self._features = {}
        self.y_flow = None
        self.x_flow = None

    def __repr__(self) -> str:
        repr_str = f"Frame time: {self._time}, "
        repr_str += f"Number of Features: {len(self._features)}"
        return repr_str

    def __eq__(self, other) -> bool:
        if not isinstance(other, Frame):
            return False
        return self.time == other.time

    @property
    def features(self) -> dict:
        """
        Get all features identified in the frame as a dict, with the feature ids as the
        dict keys and the corresponding Feature objects as the dicts vals
        """
        return self._features

    @property
    def time(self) -> dt.datetime:
        """
        Get the datetime object that the current frame is valid for
        """
        return self._time

    @property
    def feature_field(self) -> NDArray[np.integer]:
        """
        Get the feature id field for the current frame
        """
        return self._feature_field

    @property
    def lifetime_field(self) -> NDArray[np.integer]:
        """
        Get the feature lifetime field for the current frame
        """
        return self._lifetime_field

    @property
    def max_id(self) -> int:
        """
        Returns max_id of features in the frame.
        """
        return self._max_id

    @property
    def raw_field(self) -> NDArray:
        """
        Get the raw data field for the current frame
        """
        return self._raw_field

    @features.setter
    def features(self, features_dict: dict) -> None:
        """
        Set the features dict for the frame, with the feature ids as the
        dict keys and the corresponding Feature objects as the dicts vals
        """
        if not isinstance(features_dict, dict):
            raise TypeError(f"Expected type dict, got {type(features_dict)}")
        self._features = features_dict

    @time.setter
    def time(self, time: dt.datetime) -> None:
        """
        Set time for the current frame, as a datetime.datetime object
        """
        if not isinstance(time, dt.datetime):
            raise TypeError(
                f"Expected 'output_time' to be datetime objcet, got {type(time)}"
            )
        self._time = time

    @feature_field.setter
    def feature_field(self, feature_field: NDArray) -> None:
        """
        Sets the self._feature_field attribute of the frame
        """
        self._feature_field = check_arrays(feature_field, ndim=2, dtype=int)

    @lifetime_field.setter
    def lifetime_field(self, lifetime_field: NDArray) -> None:
        """
        Sets the self._lifetime_field attribute of the frame
        """
        self._lifetime_field = check_arrays(lifetime_field, ndim=2, dtype=int)

    @raw_field.setter
    def raw_field(self, raw_field: NDArray) -> None:
        """
        Sets the self._raw_field attribute of the frame
        """
        self._raw_field = check_arrays(raw_field, ndim=2)

    @max_id.setter
    def max_id(self, max_id: int) -> None:
        """
        Sets the max_id used for assigning to features that do not match to another
        feature from a previous timestep
        """
        max_id = check_valid_ids(max_id)
        self._max_id = max_id

    def get_feature(self, feature_id: int, provisional: bool = False) -> Feature:
        """
        Get a feature matching the given id if present in the current field,
        otherwise returns None.

        Args:
            feature_id (int): The id of the feature to retrieve.
            provisional (bool): If True, will search for a feature with a matching
            provisional_id instead of the id. Defaults to False.
        """
        feature_id = check_valid_ids(feature_id)
        if feature_id in self._features and not provisional:
            return self._features[feature_id]
        elif provisional:
            for feature in self._features.values():
                if feature.provisional_id == feature_id:
                    return feature
        return None

    def get_flow(self) -> NDArray | None:
        """
        Get a list of the y-flow and x-flow fields derived by comparing features between
        this frame and a frame from a previous timestep. Flow fields are both numpy
        arrays, with order [y_flow, x_flow]. If flow was not previously derived,
        returns [None, None]
        """
        return self.y_flow, self.x_flow

    def import_time_and_data(self, time: dt.datetime, data: NDArray) -> None:
        """
        Load time and raw data into the frame.

        Args:
            time (dt.datetime): Time the frame is valid for.
            data (NDArray): Raw data to perform tracking on
        """
        self._raw_field = check_arrays(data, ndim=2)
        if not isinstance(time, dt.datetime):
            raise TypeError(
                f"Expected 'output_time' to be datetime objcet, got {type(time)}"
            )
        self._time = time

    def identify_features(
        self,
        threshold: float,
        under_threshold: bool = False,
        min_size: int = 4,
    ) -> None:
        """
        Call the "label_features" function to identify distinct regions in the input
        field that meet a specified threshold condition.
        Then, analyses each of the identified features to find properties

        Args:
            - min_size (float): Minimum area (in number of grid points) for a region
            to be considered valid
            - threshold (float): Threshold value for identifying regions
            - under_threshold (bool): If True, regions under the threshold
            are considered; if False, regions over the threshold are considered.
        """
        if self._raw_field is None:
            raise Exception("Data has not been loaded into Frame")

        self._feature_field = label_features(
            field=self._raw_field,
            min_area=min_size,
            threshold=threshold,
            under_threshold=under_threshold,
        )
        # Provisionally set the lifetime field to 1 anywhere there is a feature
        self._lifetime_field = np.zeros_like(self._feature_field)
        self._lifetime_field[self._feature_field > 0] = 1
        if np.max(self._feature_field) == 0:
            return
        self.max_id = int(np.max(self._feature_field))
        self.populate_features()

    def populate_features(self) -> None:
        """
        Uses the self._feature_field array to populate the self.features dict with new
        Feature instances.
        """
        # Check for existing features dict
        if self._features:
            self._features = {}

        if self._feature_field is None:
            return

        feature_ids = np.unique(self._feature_field)
        # Remove 0 from the list of ids
        # (usually this is at idx 0 but can't be guaranteed)
        # Check whether 0 is in the feature_ids array before trying to delete it
        if 0 in feature_ids:
            feature_ids = np.delete(feature_ids, np.where(feature_ids == 0)[0][0])
        feature_ids = check_valid_ids(feature_ids)

        # Don't include 0 in Feature population, this is reserved for background
        for feature_id in feature_ids:
            # Get the pixel locations of the feature in the field
            # For 2D data, np.where returns two arrays containing y, x locations
            feature_mask = np.where(self._feature_field == feature_id)
            feature_coords = np.array(feature_mask)

            # Construct Feature object, set relevant properties,
            # add to the list of features
            feature = Feature(
                id=native(feature_id), feature_coords=feature_coords, time=self._time
            )
            # If raw field is not None, use this to find max value within Feature
            if self._raw_field is not None:
                feature.max = np.max(self._raw_field[feature_mask])
                feature.mean = np.mean(self._raw_field[feature_mask])
            self._features[native(feature_id)] = feature

    def assign_displacements(self, y_flow: NDArray, x_flow: NDArray) -> None:
        """
        Add flow field to frame. Use input y_flow and x_flow fields to assign
        dy and dx displacements to each Feature in the Frame

        Args:
            y_flow (NDArray): _description_
            x_flow (NDArray): _description_
        """
        if self._feature_field is None or not self._features:
            raise FeaturesNotFoundError(
                "Features have not been loaded into this Frame. "
                "Cannot assign displacements"
            )

        self.y_flow, self.x_flow = check_arrays(
            y_flow, x_flow, ndim=2, equal_shape=True
        )

        # Assign these displacements to each Feature in the Frame using
        # mean of flow field for each grid point spanning the Feature
        for feature_id, feature in self._features.items():
            feature_mask = self._feature_field == feature_id
            feature_dy = np.mean(y_flow[feature_mask])
            feature_dx = np.mean(x_flow[feature_mask])
            feature.dydx = (feature_dy, feature_dx)

    def get_next_available_feature_id(self) -> int:
        """
        Get the next available feature ID for this Frame.
        Used when new features are created.

        Returns:
            int: new id
        """
        if self._max_id is None:
            if self._feature_field is not None:
                self._max_id = np.max(self._feature_field).item()
            else:
                self._max_id = 0
        self._max_id += 1
        return self._max_id

    def promote_provisional_ids(self) -> None:
        """
        Promote "provisional_id" to final "id" for all features.

        Args:
            remove_provisional_id (bool, optional): Whether to remove the provisional_id
            attribute from the Feature after promotion. Defaults to True.
        """
        # Construct updated features dictionary with new ids as keys
        new_features_dict = {}

        for feature in self._features.values():
            if feature.provisional_id is not None:
                feature.id = feature.provisional_id
                feature.provisional_id = None
            new_features_dict[feature.id] = feature

        self._features = new_features_dict

    def update_fields_using_provisional_ids(self) -> None:
        """
        Update feature_field and lifetime_field using current Feature data.
        This is useful if there have been changes to the Feature data
        (e.g., Features have been matched between frames)

        Method creates a feature mask for each Feature based on its current ID and
        then updates the field to be the provisional ID
        """
        if self._feature_field is None:
            print(
                UserWarning(
                    "Feature field is not set. Cannot update using provisional ids."
                )
            )
            return

        if not self._features:
            print(
                UserWarning(
                    "Features have not been loaded into this Frame. "
                    "Cannot update using provisional ids."
                )
            )
            return

        updated_feature_field = np.zeros_like(self._feature_field)
        updated_lifetime_field = np.zeros_like(self._feature_field)

        for feature in self._features.values():
            feature_mask = self._feature_field == feature.id
            updated_lifetime_field[feature_mask] = feature.lifetime
            if feature.provisional_id is not None:
                updated_feature_field[feature_mask] = feature.provisional_id
            else:
                updated_feature_field[feature_mask] = feature.id

        self._feature_field = updated_feature_field
        self._lifetime_field = updated_lifetime_field

    def get_new_features(self) -> list:
        """
        Get a list of all features in the frame that do not match with a feature from
        the previous frame and has not split from a feature in the previous frame
        """
        if not self._features:
            return []
        return [feature for feature in self._features.values() if feature.is_new()]

    def get_dissipating_features(self) -> list:
        """
        Get a list of all features in the frame that do not match with a feature
        in the subsequent frame and have not merged with a feature
        in the subsequent frame
        """
        if not self._features:
            return []
        return [
            feature for feature in self._features.values() if feature.is_dissipating()
        ]

    def get_init_field(self, centroid_only: bool = False) -> NDArray:
        """
        Get a binary field of locations where features are newly initialising, where
        new features are ones that are not matched with a feature in the previous frame,
        and have not split from a feature in the previous frame
        """
        return self.get_field("init", centroid_only)

    def get_dissipation_field(self, centroid_only: bool = False) -> NDArray:
        """
        Get a binary field of locations where features are dissipating, where
        these are ones that are not matched with a feature in the next frame, and
        do not merge with a feature in the next frame
        """
        return self.get_field("dissipation", centroid_only)

    def get_field(self, field_type: str, centroid_only: bool = True) -> NDArray:
        """
        Get a binary field of locations where features meet the input requirement,
        as speicified by field type

        Args:
            field_type (str):
                "init": Get the field of all new features in the frame, where new
                features are ones that are not matched with a feature in the previous
                frame, and have not split from a feature in the previous frame
                "dissipation" Get the fields of all dissipating feature in the frame,
                where these are ones that are not matched with a feature in the next
                frame, and do not merge with a feature in the next frame
            centroid_only (bool, optional):
                Whether the binary output should contain just the feature centroids
                or should span the full feature shape.
                Defaults to True.

        """
        feature_methods = {
            "init": self.get_new_features,
            "dissipation": self.get_dissipating_features,
        }
        if field_type not in feature_methods:
            raise KeyError(f"field_type must be one of {feature_methods.keys()}")

        field = np.zeros_like(self._feature_field)
        for feature in feature_methods[field_type]():
            if centroid_only:
                # tuple to ensure correct indexing
                # Round centroid to nearest integer and cast to int
                centroid_coord = tuple(np.rint(feature.centroid).astype(int))
                field[centroid_coord] = 1
            else:
                # Populate field with full size of feature
                init_mask = self._feature_field == feature.id
                field[init_mask] = 1
        return field

    def plot(
        self,
        field: str = "id",
        ax: plt.Axes = None,
        highlight_features: list | int = None,
        highlight_at_coords: list[tuple] = None,
        vmax: int = None,
        alpha: float = 1.0,
        show_cbar: bool = True,
    ) -> tuple:
        """
        Plot a field (id, lifetime, or motion vectors) of the current Frame.
        Can highlight a given Feature or list of Features. Can also plot on an
        existing axis if provided.

        Args:
            field (str, optional):
                Field to plot. Can be "id", "lifetime", or "motion".
                Defaults to "id".
            ax (plt.Axes, optional):
                The axis on which to plot. If None, a new axis will be created.
                Defaults to None.
            highlight_features (list | int, optional):
                The Feature or list of Features to highlight.
                Defaults to None.
            highlight_at_coords (list[tuple], optional):
                A list of (y, x) coordinates which will highlight any features
                that overlap with those coordinates. Defaults to None.
            vmax (int, optional):
                The maximum value for the colorbar. If None, will be set to the max
                value of the field being plotted. Defaults to None.
            alpha (float, optional):
                The alpha value for the non-highlighted features. Defaults to 1.0.


        Returns:
            fig, ax: plt Figure and Axes objects of the plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Map field string to actual field data
        field_map = {
            "id": {
                "data": self._feature_field,
                "cbar": plt.get_cmap("viridis"),
                "cbar_label": "Feature ID",
            },
            "lifetime": {
                "data": self._lifetime_field,
                "cbar": plt.get_cmap("YlGnBu"),
                "cbar_label": "Lifetime",
            },
            "motion": {
                "data": (self.y_flow, self.x_flow),
                "cbar": None,
                "cbar_label": None,
            },
        }

        if field not in field_map:
            raise ValueError(f"Field must be one of {list(field_map.keys())}")

        field_data = field_map[field]["data"]
        cbar = field_map[field]["cbar"]
        cbar_label = field_map[field]["cbar_label"]

        # If cbar is set, plot filled contours
        # Otherwise, plot quiver for motion vectors
        if cbar is not None:
            pcm = self.__plot_filled_contours(ax, field_data, cbar, vmax, alpha=alpha)
            if show_cbar:
                plt.colorbar(pcm, ax=ax, label=cbar_label)
        else:
            self.__plot_quiver(ax, field_data[0], field_data[1])

        # Add feature contour outlines
        binary_field = np.where(self._feature_field > 0, 1, 0)
        ax.contour(binary_field, colors="teal", linewidths=0.1, alpha=alpha)

        # Get list of all features to highlight, either from input list or
        # from coordinates
        features_to_highlight = []
        if highlight_features is not None:
            if isinstance(highlight_features, int):
                highlight_features = [highlight_features]
            features_to_highlight.extend(highlight_features)

        if highlight_at_coords is not None:
            for coord in highlight_at_coords:
                y, x = coord
                if (
                    y < 0
                    or y >= self._feature_field.shape[0]
                    or x < 0
                    or x >= self._feature_field.shape[1]
                ):
                    print(f"Coordinate {coord} is out of bounds for the feature field")
                    continue
                feature_id = self._feature_field[y, x]
                if feature_id > 0:
                    features_to_highlight.append(feature_id)
                else:
                    print(f"No feature found at coordinate {coord}")

        if len(features_to_highlight) > 0:
            for feature_id in features_to_highlight:
                if feature_id not in self._features:
                    print(f"Feature {feature_id} not found in Frame")
                    continue
                feature = self._features[feature_id]
                centroid = feature.centroid
                ax.contour(
                    self._feature_field == feature_id,
                    colors="k",
                    linewidths=1.5,
                )
                ax.annotate(feature_id, xy=centroid[::-1], color="k", fontsize=8)

        return fig, ax

    def __plot_filled_contours(
        self,
        ax: plt.Axes,
        field_data: NDArray,
        cmap: plt.cm,
        vmax: int,
        alpha: float = 1.0,
    ) -> None:
        """
        Internal method to plot filled contours.
        """
        cmap = cmap.with_extremes(under="white")
        pcm = ax.pcolormesh(field_data, cmap=cmap, vmin=1, vmax=vmax, alpha=alpha)
        return pcm

    def __plot_quiver(
        self, ax: plt.Axes, y_flow: NDArray, x_flow: NDArray, step: int = 20
    ) -> None:
        """
        Internal method to plot quiver for motion vectors.
        """
        if y_flow is None or x_flow is None:
            print("Flow fields are not set. Cannot plot motion vectors.")
            return
        xmesh, ymesh = np.meshgrid(
            np.arange(y_flow.shape[1]), np.arange(y_flow.shape[0])
        )
        ax.quiver(
            xmesh[::step, ::step],
            ymesh[::step, ::step],
            x_flow[::step, ::step],
            y_flow[::step, ::step],
            color="k",
            alpha=0.5,
        )


class Timeline:
    """
    Object for storing and accessing Frames, stored as a dict of dt.datetime keys
    and Frame values.
    """

    def __init__(self, max_frames: int = None) -> None:
        """
        Setup the Timeline object.

        Args:
            max_frames (int, optional):
                The maximum number of frames to keep in the timeline.
                Defaults to None (i.e., keep all frames)
        """
        self.timeline = {}
        self.max_frames = max_frames

    def __repr__(self) -> str:
        repr_str = (
            f"Timeline with {len(self.timeline)} frames, "
            f"starting at {self.start_time()} "
            f"ending at {self.end_time()}"
        )
        return repr_str

    def start_time(self) -> dt.datetime:
        """
        Get the start datetime of the timeline
        """
        if len(self.timeline) == 0:
            return None
        return min(self.timeline.keys())

    def end_time(self) -> dt.datetime:
        """
        Get the end datetime of the timeline.
        """
        if len(self.timeline) == 0:
            return None
        return max(self.timeline.keys())

    def get_start_frame(self) -> Frame:
        """
        Get the first frame in the timeline.
        """
        start_time = self.start_time()
        if start_time is None:
            return None
        return self.timeline[start_time]

    def get_end_frame(self) -> Frame:
        """
        Get the last frame in the timeline.
        """
        end_time = self.end_time()
        if end_time is None:
            return None
        return self.timeline[end_time]

    def __len__(self) -> int:
        return len(self.timeline)

    def add_to_timelime(self, frame: Frame) -> None:
        """
        Add the input frame to the timeline, using the frame.get_time() to
        determine the frame time.
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"Expected type Frame, got {type(frame)}")
        if frame.time is None:
            raise ValueError("Frame time is not set. Cannot add to timeline.")
        self.timeline[frame.time] = frame

        # Check if older frames need to be purged
        if self.max_frames is not None and len(self.timeline) > self.max_frames:
            self.purge_old_frame()

    def get_previous_frame(self, current_time: dt.time) -> Frame:
        """
        Finds the frame with the closest time to the input frame, and which
        is in the past.
        """
        if len(self.timeline) == 0:
            raise ValueError("Timeline is empty. No previous frame to return.")
        if len(self.timeline) == 1:
            return None

        prev_times = [time for time in self.timeline if time < current_time]
        closest_time = max(prev_times) if prev_times else None
        if closest_time is None:
            raise ValueError("No previous frame found in timeline")
        return self.timeline[closest_time]

    def purge_old_frame(self) -> None:
        # Remove any frames that aren't needed anymore, as defined by self.max_frames
        frames_to_remove = len(self.timeline) - self.max_frames
        if frames_to_remove > 0:
            # Sort the timeline by time and remove the oldest frames
            sorted_times = sorted(self.timeline.keys())
            times_to_remove = sorted_times[:frames_to_remove]
            for time in times_to_remove:
                del self.timeline[time]

    def get_timeline(self) -> dict:
        """
        Return the timeline as a dictionary of values, with keys being the validity
        time and values being the frame at that validity time.
        """
        return self.timeline

    def get_frame(self, time: dt.datetime) -> Frame:
        """
        Get the frame that is valid at the input time. Raises ValueError if frame
        matching the input time is not found.
        """
        if time not in self.timeline:
            raise ValueError(f"No frame found for time {time}")
        return self.timeline[time]

    def plot_trajectories(
        self,
        latest_frame_time: dt.datetime,
        features: int | list,
        show_centroid: bool = True,
        show_outline: bool = False,
        ax: plt.Axes = None,
    ) -> tuple:
        """
        Plot the trajectories of the features in the timeline up to the input frame
        time. If features is specified, only plot those features.

        Args:
            latest_frame_time (dt.datetime):
                The time of the frame to plot current features
            features (int | list, optional):
                The feature(s) to plot trajectories for
            show_centroid (bool, optional):
                Whether to show the centroid of the features in each frame.
                Defaults to True.
            show_outline (bool, optional):
                Whether to show the outline of the features in each frame.
                Defaults to False.
            ax (plt.Axes, optional):
                The axis to plot on. If None, create a new axis.
            Defaults to None.
        """
        current_frame = self.get_frame(latest_frame_time)

        # Plot all features in current frame for context
        fig, ax = current_frame.plot(
            field="id", ax=ax, highlight_features=features, alpha=0.3
        )

        if isinstance(features, int):
            features = [features]

        for feature_id in features:
            # Get all frames before the latest frame time where the feature exists
            feature_frames = [
                frame
                for frame in self.timeline.values()
                if feature_id in frame.features and frame.time <= latest_frame_time
            ]
            if not feature_frames:
                print(f"Feature {feature_id} not found in any frames")
                continue

            # Plot the trajectory of the feature
            prev_centroid = None
            for frame in feature_frames:
                # # Plot the feature outline
                if show_outline:
                    feature_mask = frame.feature_field == feature_id
                    ax.contour(feature_mask, colors="k", linewidths=1.5)

                # Plot feature centroid and arrow between previous and current centroid
                if show_centroid:
                    # Get centroid of the feature in this frame
                    feature = frame.get_feature(feature_id)
                    centroid = feature.centroid
                    ax.scatter(centroid[1], centroid[0], color="k", s=20, marker="x")

                    # If there is a previous centroid, plot a line from
                    # the previous centroid to the current centroid
                    if prev_centroid is not None:
                        ax.plot(
                            [prev_centroid[1], centroid[1]],
                            [prev_centroid[0], centroid[0]],
                            color="k",
                            linewidth=1.5,
                        )
                    prev_centroid = centroid

        return fig, ax

    def animate(
        self,
        field: str = "id",
        interval: int = 500,
        highlight_features: list = None,
        highlight_at_coords: list = None,
        save_as_gif: bool = False,
    ) -> animation.FuncAnimation:
        """
        Animate the timeline of frames, showing the specified field.

        Args:
            field (str, optional):
                The field to animate. Can be "id", "lifetime", or "motion".
                Defaults to "id".
            interval (int, optional):
                The interval between frames in milliseconds. Defaults to 500.
            highlight_features (list, optional):
                A list of feature IDs to highlight in the animation. Defaults to None.
            highlight_at_coords (list, optional):
                A list of (y, x) coordinates to highlight features at in the animation.
                Defaults to None.
            save_as_gif (bool, optional):
                Whether to save the animation as a GIF. Defaults to False.
        """
        print(
            "To view this animation in a Jupyter notebook, use the following command:"
        )
        print("from IPython.display import HTML")
        print("anim = timeline.animate(...)")
        print("HTML(anim.to_jshtml())")

        # Get vmax for consistent color scaling across frames
        if field == "id":
            vmax = self.get_end_frame().max_id
        elif field == "lifetime":
            vmax = np.max(self.get_end_frame().lifetime_field)
        else:
            vmax = None

        fig, ax = plt.subplots()

        frame_times = sorted(self.timeline.keys())
        colorbar = None

        if field in ("id", "lifetime"):
            first_frame = self.get_frame(frame_times[0])
            first_frame.plot(field=field, ax=ax, vmax=vmax, show_cbar=False)
            colorbar = fig.colorbar(ax.collections[0], ax=ax, label=field.title())

        def update(frame_time):
            ax.clear()
            frame = self.get_frame(frame_time)
            frame.plot(
                field=field,
                ax=ax,
                highlight_features=highlight_features,
                highlight_at_coords=highlight_at_coords,
                vmax=vmax,
                show_cbar=False,
            )
            if colorbar is not None:
                colorbar.update_normal(ax.collections[0])

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=sorted(self.timeline.keys()),
            interval=interval,
            repeat=True,
            repeat_delay=5000,
        )
        if save_as_gif:
            ani.save(
                "timeline_animation.gif",
                writer=animation.PillowWriter(fps=1000 / interval),
            )
        return ani


def label_features(
    field: NDArray[np.floating],
    min_area: float,
    threshold: float,
    under_threshold: bool = False,
    connectivity_structure: str = "default",
) -> NDArray[np.integer]:
    """
    Label distinct regions in the input field that meet a specified threshold condition.

    Args:
        field (np.ndarray):
            2D input array of data to be labelled
        min_area (float):
            Minimum area (in number of grid points) for a region to be considered valid
        threshold (float):
            Threshold value for identifying regions
        under_threshold (bool, optional):
            If True, regions under the threshold are considered;
            if False, regions over the threshold are considered.
            Defaults to False.
        connectivity_structure (str, optional):
            Str defining boolean connectivity for region labelling
            Default is 8-way connectivity, meaning all cardinal AND diagonal neighbours
            that meet the threshold condition are considered part of the same region.
            An alternative arrangement would be "cardinal" 4-way connectivity
            (diagonals omitted), defined as:
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
            See scipy.ndimage.label documentation for more details.
            Defaults to np.ones((3, 3)).

    Raises:
        TypeError: field must be a numpy ndarray
        ValueError: min_area must be a non-negative number
        ValueError: threshold must be a number"
        TypeError: under_threshold must be a boolean
        ValueError: field must be a 2D array

    Returns:
        NDArray[np.int_]: 2D Integer field of labelled regions,
        same shape as input field
    """

    # Check input types
    field = check_arrays(field, ndim=2)

    # Handle isntance of MaskedArray by filling any masked areas with 0 (background)
    if isinstance(field, np.ma.MaskedArray):
        field = field.filled(fill_value=0)

    if not isinstance(min_area, (int, float)) or min_area < 0:
        raise ValueError("min_area must be a non-negative number")
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if not isinstance(under_threshold, bool):
        raise TypeError("under_threshold must be a boolean")

    if connectivity_structure == "default":
        # All cardinal and diagonal points
        connectivity_structure = np.ones((3, 3))
    else:
        raise NotImplementedError("Only default connectivity supported")

    # Construct feature field using threshold and threshold condition
    # Grid points meeting the condition are set to 1, others to 0s
    if under_threshold:
        feature_field = np.where(field < threshold, 1, 0)
    else:
        feature_field = np.where(field > threshold, 1, 0)

    # Identify and label distinct regions in the feature field
    id_regions, num_ids = ndimage.label(feature_field, structure=connectivity_structure)

    # Any regions smaller than the min_area are removed from the feature field
    # before re-running feature labelling
    id_sizes = np.array(ndimage.sum(feature_field, id_regions, range(num_ids + 1)))
    area_mask = id_sizes < min_area
    feature_field[area_mask[id_regions]] = 0
    id_regions, num_ids = ndimage.label(feature_field, structure=connectivity_structure)

    return id_regions
