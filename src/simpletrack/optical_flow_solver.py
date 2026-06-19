import cv2
import numpy as np

from simpletrack.frame import Frame


class DISFlowSolver:
    """
    Class containing methods for calculating optical flow between two frames using
    OpenCV's DIS optical flow algorithm [Kroeger et al., 2016].
    """

    def __init__(self, patch_size=50):
        """
        Initialize the DISFlowSolver.

        Args:
            patch_size (int, optional):
            Size of the patch used for matching. Larger values result in smoother
            flow but may miss small details. Defaults to 50.
        """
        self.patch_size = patch_size
        print(self.patch_size)

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

        # Create an instance of the DIS optical flow algorithm
        # Use the "2" preset, which is slower but more accurate
        of_instance = cv2.DISOpticalFlow_create(2)
        of_instance.setPatchSize(self.patch_size)

        # Calculate the optical flow between the two frames
        flow = of_instance.calc(prev_features, current_features, None)
        x_flow, y_flow = flow[..., 0], flow[..., 1]

        return y_flow, x_flow
