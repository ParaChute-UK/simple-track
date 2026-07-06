import numpy as np
import skimage

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

    def analyse_flow(self, prev_field, current_field):
        if isinstance(prev_field, Frame) and isinstance(current_field, Frame):
            prev_features = prev_field.feature_field.astype(np.uint8)
            current_features = current_field.feature_field.astype(np.uint8)
        elif isinstance(prev_field, np.ndarray) and isinstance(
            current_field, np.ndarray
        ):
            prev_features = prev_field.astype(np.uint8)
            current_features = current_field.astype(np.uint8)
        else:
            raise TypeError(
                "prev_field and current_field must both be of type Frame or NDArray"
            )

        # Run optical flow using the TV-L1 algorithm from scikit-image
        y_flow, x_flow = skimage.registration.optical_flow_tvl1(
            prev_features,
            current_features,
            attachment=self.attachment,
            tightness=self.tightness,
        )

        return y_flow, x_flow
