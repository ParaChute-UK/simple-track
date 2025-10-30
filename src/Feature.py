from numpy.typing import NDArray
import numpy as np


class Feature:
    def __init__(self, id: int, feature_coords: NDArray[np.integer]):
        if feature_coords.shape[0] != 2 or feature_coords.ndim != 2:
            exc_str = "Expected a 2D array with first dimension of size 2 "
            exc_str += f"got {feature_coords.ndim} array with first dimension"
            exc_str += f"of size {feature_coords.shape[0]}"
            raise TypeError(exc_str)

        self.id = id
        self.centroid = None
        self.feature_coords = feature_coords
        self.lifetime = 1
        self.accreted = -999
        self.parent = -999
        self.child = -999
        self.accreted = -999
        self.dx = 0
        self.dy = 0


class RadarFeature(Feature):
    pass
