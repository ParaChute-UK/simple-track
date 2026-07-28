import numpy as np

from simpletrack.frame import Frame


class DISFlowSolver:
    """
    Class containing methods for calculating optical flow between two frames using
    OpenCV's DIS optical flow algorithm [Kroeger et al., 2016].
    """

    def __init__(self, subdomain_size=None):
        """
        Initialize the DISFlowSolver.

        Args:
            subdomain_size (int, optional):
                Size of the subdomain used for matching.
                Defaults to feature_field.shape // 5 if not provided.
        """
        self.patch_size = subdomain_size

    def analyse_flow(self, prev_field, current_field):
        # Set import here, since this is an optional dependency
        import cv2

        # Extract field and normalize to unsigned 8-bit for optical flow calculation
        if isinstance(prev_field, Frame) and isinstance(current_field, Frame):
            prev_features = normalize8(prev_field.feature_field)
            current_features = normalize8(current_field.feature_field)
        elif isinstance(prev_field, np.ndarray) and isinstance(
            current_field, np.ndarray
        ):
            prev_features = normalize8(prev_field)
            current_features = normalize8(current_field)
        else:
            raise TypeError(
                "prev_field and current_field must both be of type Frame or NDArray"
            )

        # Check inputs are not empty
        if not np.count_nonzero(prev_features) or not np.count_nonzero(
            current_features
        ):
            print("No features detected in an input field. Skipping optical flow.")
            return None, None

        if self.patch_size is None or self.patch_size == "default":
            self.patch_size = self.get_patch_size(prev_features.shape)

        # Create an instance of the DIS optical flow algorithm
        # Use the "2" preset, which is slower but more accurate
        of_instance = cv2.DISOpticalFlow_create(2)
        of_instance.setPatchSize(self.patch_size)

        # Calculate the optical flow between the two frames
        flow = of_instance.calc(prev_features, current_features, None)
        x_flow, y_flow = flow[..., 0], flow[..., 1]

        return y_flow, x_flow

    def check_cv2_importable(self) -> bool:
        """
        Check if OpenCV is importable.

        Returns:
            bool: True if OpenCV is importable, False otherwise.
        """
        try:
            import cv2  # noqa: F401

            return True
        except ImportError:
            return False

    def get_patch_size(self, field_shape: tuple) -> int:
        """
        Get the patch size for the DIS optical flow algorithm based on the shape of the
        input field.

        Args:
            field_shape (tuple):
                Shape of the input field.

        Returns:
            int: Patch size for the DIS optical flow algorithm.
        """
        min_dim = min(field_shape)
        patch_size = max(5, min_dim // 5)  # Ensure patch size is at least 5 pixels
        return patch_size


def normalize8(array: np.ndarray) -> np.ndarray:
    mn = array.min()
    mx = array.max()

    mx -= mn

    array = ((array - mn) / mx) * 255
    return array.astype(np.uint8)
