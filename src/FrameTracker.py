import numpy as np
from numpy.typing import NDArray

from Frame import Frame


class FrameTracker:
    def __init__(
        self,
        overlap_nbhood: int = 5,
        overlap_threshold: float = 0.6,
    ):
        self.overlap_nbhood = overlap_nbhood
        self.overlap_threshold = overlap_threshold

    def run(self, prev_frame: Frame, current_frame: Frame) -> None:
        if not all(isinstance(frame, Frame) for frame in [prev_frame, current_frame]):
            raise TypeError(
                f"Expected type Frame, got {type(prev_frame)} and {type(current_frame)}"
            )

        # Advect features in the previous frame using its flow field
        advected_frame = self.advect_frame(prev_frame)

        # Get the feature fields to analyse
        advected_feature_field = advected_frame.get_feature_field()
        current_feature_field = current_frame.get_feature_field().copy()

        # Compare features in the advected frame and current frame by looping over
        # all features in current frame
        for feature_id, current_feature in current_frame.get_features().items():
            # Count the ids contained in advected_feature_field that are in the same
            # position as the feature_id in the current_feature_field (normalised by
            # the feature sizes)
            overlap_hist = self.calculate_overlap_histogram(
                advected_feature_field, current_feature_field, feature_id, nbhood=None
            )

            # If the maximum overlap is not achieved, rerun with a nbhood surrouding the
            # feature centroid. Only look from index 1 onwards since 0 is treated as the background
            max_overlap = np.max(overlap_hist[1:])
            if max_overlap < self.overlap_threshold:
                overlap_hist = self.calculate_overlap_histogram(
                    advected_feature_field,
                    current_feature_field,
                    feature_id,
                    nbhood=self.overlap_nbhood,
                )

            # Now, check number of sufficient overlaps.
            # len_overlaps = overlap_hist[1:] >=
            # If there are no overlaps, this is a new Feature and needs a new id

    def calculate_overlap_histogram(
        self,
        advected_feature_field: NDArray,
        current_feature_field: NDArray,
        feature_id: int,
        nbhood: int = 0,
    ) -> NDArray:
        """
        Calculate the amount of overlap between two feature fields at the requested feature id.
        Method creates a mask containing areas of the current feature field containing the
        requested feature id. This mask can optionally be expanded using a nbhood surrouding the
        centroid of this feature. This mask is then used to select the same locations of the
        advected feature field. A histogram is produced giving the number of feature_ids contained
        in this reigon in the adveced feature field. This is normalised by the pixel size of the
        feature in each input field.

        Args:
            advected_feature_field (NDArray): _description_
            current_feature_field (NDArray): _description_
            feature_id (int): _description_
            nbhood (bool, optional): _description_. Defaults to False.

        Returns:
            NDArray: _description_
        """
        # Create feature mask using current feature field, or expand mask using a nbhood
        # if this is flagged in input
        feature_mask = np.where(current_feature_field == feature_id, True, False)
        if nbhood:
            centroid = get_centroid(current_feature_field, feature_id)
            feature_mask += generate_radial_mask(
                current_feature_field, centroid, nbhood
            )

        # Setup bins for comparing feature fields using histogram
        # TODO: is it important whether advected or current feature field is used here?
        bins = np.arange(int(np.max(advected_feature_field)) + 2)

        # Find overlap between two feature fields by finding histogram of points
        # using mask of current features applied to the advected feature field
        overlap_hist = np.histogram(advected_feature_field[feature_mask], bins)[0]

        # Get size (number of pixels) of each feature for normalisation
        advected_feature_size = np.count_nonzero(advected_feature_field == feature_id)
        current_feature_size = np.count_nonzero(current_feature_field == feature_id)

        overlap_normed = overlap_hist / advected_feature_size
        overlap_normed += overlap_hist / current_feature_size
        return overlap_normed / 2

    def advect_frame(self, frame: Frame) -> Frame:
        """
        Construct a new Frame with all Features in the input Frame advected by the given flow field

        Args:
            frame (Frame): Frame containing Features and a flow field

        Returns:
            Frame: advected Frame
        """

        if not isinstance(frame, Frame):
            raise TypeError(f"Expected 'Frame', got {type(frame)}")

        y_flow, x_flow = frame.get_flow()
        if y_flow is None or x_flow is None:
            print(f"y_flow: {y_flow}")
            print(f"x_flow: {x_flow}")
            raise ValueError(
                "y_flow and/or x_flow not defined. Cannot calculate advected Frame"
            )

        feature_field = frame.get_feature_field()
        advected_feature_field = advect_field_using_motion_vectors(
            feature_field, y_flow, x_flow
        )

        advected_frame = Frame()
        advected_frame.set_time(frame.get_time())
        advected_frame.set_feature_field(advected_feature_field)
        advected_frame.populate_features()


def advect_field_using_motion_vectors(
    field: NDArray, y_flow: NDArray, x_flow: NDArray
) -> NDArray:
    """
    A (perhaps temporary) function that takes features (non-zero elements) in the 2D input field
    and advects them using the given motion vectors. This function performs this advection feature
    by feature, i.e., moves all contiguous elements of data by the same amount, retaining their shape.
    Code handles conflicts from multiple non-zero elements being advected to the same position by
    choosing the closest label centroid.

    Args:
        field (NDArray): _description_
        y_flow (NDArray): _description_
        x_flow (NDArray): _description_

    Returns:
        NDArray: _description_
    """
    if not all((isinstance(arg, np.ndarray) for arg in [field, y_flow, x_flow])):
        raise TypeError(
            f"Expected NDArray, got {type(field)}, {type(y_flow)}, {type(x_flow)}"
        )

    if not all(arg.ndim == 2 for arg in [field, y_flow, x_flow]):
        raise ValueError("Expected 2D input fields for all args")

    # TODO: check equal input shapes

    advected_field = np.zeros_like(field)

    # Loop over all features (non-zero elements) in field and advect by mean flow across the feature
    # Background field is assumed to be 0
    for feature_id in range(1, np.max(field) + 1):
        feature_mask = np.where(field == feature_id)
        dy = np.mean(y_flow[feature_mask]).astype(int)
        dx = np.mean(x_flow[feature_mask]).astype(int)

        # Now, advect the feature to the new position
        for y_coord, x_coord in zip(*feature_mask):
            advected_y_coord = y_coord + dy
            advected_x_coord = x_coord + dx

            # If this coordinate is out of bounds of the field, no further action needed
            oob_y_check = advected_y_coord < 0 or advected_y_coord > field.shape[0] - 1
            oob_x_check = advected_y_coord < 0 or advected_x_coord > field.shape[1] - 1
            if oob_y_check or oob_x_check:
                continue

            # If there is no label already at this position, can go ahead and place this feature
            # label here.
            id_at_coord = advected_field[advected_y_coord, advected_x_coord]
            if id_at_coord == 0:
                advected_field[advected_y_coord, advected_x_coord] = feature_id
                continue

            # Otherwise, need to handle conflicting features. Do this by finding distances between
            # the advected coordinate and the centroids of existing and iterating ids.
            # Choose the feature that is closest to its centroid
            existing_id_centroid = get_centroid(field, id_at_coord)
            iterating_id_centroid = get_centroid(field, feature_id)
            advected_coords = np.array([advected_y_coord, advected_x_coord])
            existing_id_centroid_distance = np.linalg.norm(
                existing_id_centroid - advected_coords
            )
            iterating_id_centroid_distance = np.linalg.norm(
                iterating_id_centroid - advected_coords
            )

            if iterating_id_centroid_distance < existing_id_centroid_distance:
                advected_field[advected_y_coord, advected_x_coord] = feature_id

    return advected_field


def generate_radial_mask(field: NDArray, coord: NDArray, mask_radius: int) -> NDArray:
    """
    Creates a radial mask of the same shape as the input field, centered on the (y,x) coord
    with radius equal to the mask radius.

    Args:
        field (NDArray):
            Field to generate mask for. Output will be the same shape
        coord (NDArray):
            Coordinate to centre the mask on
        mask_radius (int):
            Radius of values to include in circular mask.

    Returns:
        NDArray: Mask of values.
    """
    temp_y = np.arange(field.shape[0])
    temp_x = np.arange(field.shape[1])

    y_centroid_dist = (temp_y[:, np.newaxis] - coord[0]) ** 2
    x_centroid_dist = (temp_x[np.newaxis, :] - coord[1]) ** 2
    mask = (y_centroid_dist + x_centroid_dist) < mask_radius**2
    return mask


# TODO: replace this with the centroid calculation function in Feature?
def get_centroid(field: NDArray, value: int) -> NDArray:
    """
    From an input field, get the centroid location of a value of contiguous data

    Args:
        field (NDArray): _description_
        value (int): _description_

    Returns:
        NDArray: _description_
    """

    if not isinstance(field, np.ndarray):
        raise TypeError(f"Expected NDArray, got {type(field)}")

    if not np.issubdtype(type(value), np.integer):
        raise TypeError(f"Expected int, got {type(value)}")

    if not field.ndim == 2:
        raise ValueError("Expected 2D input fields")

    value_coords = np.where(field == value)
    centroid = np.mean(value_coords, axis=1)
    return centroid
