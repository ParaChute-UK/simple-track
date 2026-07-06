import numpy as np
import skimage
from numpy.typing import NDArray

from simpletrack.frame import Frame


class TVL1FlowSolver:
    """
    Class containing methods for calculating optical flow between two frames using
    scikit image tvl1 optical flow algorithm [Zach et al., 2016].
    https://scikit-image.org/docs/stable/api/skimage.registration.html#module-skimage.registration
    """

    def __init__(self, attachment=0.7, tightness=0.3):
        """
        Initialize the TVL1FlowSolver.
        Default values for attachment and tightness were found to minimise RMSE when
        predicting the location of features in a given frame using its previous and next
        frames for estimating the flow field.

        Args:
            attachment (float):
                Attachment parameter for the TV-L1 algorithm.
                Smaller values return smoother flow fields.
                Defaults to 0.7
            tightness (float):
                Tightness parameter for the TV-L1 algorithm, which determines
                trade off between regularisation (smoothness) and data attachement
                (accuracy to the input images). Smaller values return smoother
                flow fields.
                Defaults to 0.3
        """
        if attachment == "default":
            attachment = 0.7
        if tightness == "default":
            tightness = 0.3

        self.attachment = attachment
        self.tightness = tightness

    def analyse_flow(
        self, prev_field: Frame | NDArray, current_field: Frame | NDArray
    ) -> tuple[NDArray, NDArray]:
        if isinstance(prev_field, Frame) and isinstance(current_field, Frame):
            prev_features = prev_field.raw_field
            current_features = current_field.raw_field
        elif isinstance(prev_field, np.ndarray) and isinstance(
            current_field, np.ndarray
        ):
            prev_features = prev_field
            current_features = current_field
        else:
            raise TypeError(
                "prev_field and current_field must both be of type Frame or NDArray"
            )

        # TV-L1 expects image-like float inputs. Feature fields are integer labels,
        # so convert to binary float fields to avoid dtype/range scale issues.
        prev_features = prev_features.astype(np.float32)
        current_features = current_features.astype(np.float32)

        # Check inputs are not empty
        if not np.count_nonzero(prev_features) or not np.count_nonzero(
            current_features
        ):
            print("No features detected in an input field. Skipping optical flow.")
            return None, None

        # Run optical flow using the TV-L1 algorithm from scikit-image
        y_flow, x_flow = skimage.registration.optical_flow_tvl1(
            prev_features,
            current_features,
            attachment=self.attachment,
            tightness=self.tightness,
            num_warp=5,
        )

        return y_flow, x_flow
