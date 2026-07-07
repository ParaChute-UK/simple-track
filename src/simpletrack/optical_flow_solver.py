import numpy as np
import skimage
from numpy.typing import NDArray

from simpletrack.frame import Frame
from simpletrack.utils import check_arrays


class ILKFlowSolver:
    """
    Class containing methods for calculating optical flow between two frames using
    scikit image ILK optical flow algorithm [Zach et al., 2016].
    https://scikit-image.org/docs/stable/api/skimage.registration.html#module-skimage.registration
    """

    def __init__(self, radius: int = None):
        """
        Initialize the ILKFlowSolver.
        Default values for radius were found to minimise RMSE when
        predicting the location of features in a given frame using its previous and next
        frames for estimating the flow field.

        Args:
            radius (int, optional):
                Radius of the local window used for computing the optical flow.
                Defaults to None, which will set the radius to 1/5 of
                the smaller dimension of the input arrays.
        """
        if radius == "default":
            radius = None
        self.radius = radius

    def analyse_flow(
        self, prev_field: Frame | NDArray, current_field: Frame | NDArray
    ) -> tuple[NDArray, NDArray]:
        # Extract and convert data to float32 type
        if isinstance(prev_field, Frame) and isinstance(current_field, Frame):
            prev_features = prev_field.raw_field.astype(np.float32)
            current_features = current_field.raw_field.astype(np.float32)
        elif isinstance(prev_field, np.ndarray) and isinstance(
            current_field, np.ndarray
        ):
            prev_features = prev_field.astype(np.float32)
            current_features = current_field.astype(np.float32)
        else:
            raise TypeError(
                "prev_field and current_field must both be of type Frame or NDArray"
            )

        # Check input arrays
        prev_features, current_features = check_arrays(
            prev_features,
            current_features,
            ndim=2,
            equal_shape=True,
        )

        # Check inputs are not empty
        if not np.count_nonzero(prev_features) or not np.count_nonzero(
            current_features
        ):
            print("No features detected in an input field. Skipping optical flow.")
            return None, None

        # Calculate radius if not defined
        if self.radius is None:
            self.radius = min(prev_features.shape) // 5

        # Run optical flow using the ILK algorithm from scikit-image
        y_flow, x_flow = skimage.registration.optical_flow_ilk(
            prev_features,
            current_features,
            radius=self.radius,
        )

        return y_flow, x_flow
