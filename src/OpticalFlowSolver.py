import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from Frame import Frame
from typing import Union
from utils import check_arrays
import itertools
from skimage.registration import phase_cross_correlation
from scipy.signal.windows import tukey


class OpticalFlowSolver:
    def __init__(
        self,
        subdomain_size=None,
        min_fractional_coverage=0.01,
        dd_tolerance=3,
        overlap_threshold=0.6,
    ) -> None:
        self.subdomain_size = subdomain_size
        self.min_fractional_coverage = min_fractional_coverage
        self.dd_tolerance = dd_tolerance
        self.overlap_threshold = overlap_threshold
        self.apply_tukey_filtering = True

    def _check_inputs(self, arr1: NDArray, arr2: NDArray) -> bool:
        # Check both fields have features
        if not np.count_nonzero(arr1) and not np.count_nonzero(arr2):
            print("No features detected in both fields. Skipping optical flow.")
            return False, False

        # If there are too few features, don't proceed with optical flow
        min_feature_coverage = self.subdomain_size**2 * self.min_fractional_coverage
        if np.sum(arr1) < min_feature_coverage or np.sum(arr2) < min_feature_coverage:
            print(f"Threshold for running optical flow: {self.fftpixels}")
            print(f"Number of pixels above treshold in arr1: {np.sum(arr1)}")
            print(f"Number of pixels above treshold in arr2: {np.sum(arr2)}")
            print("Number of features in arr1 and/or arr2 less than threshold. ")
            print("Skipping optical flow")
            return False, False

        return arr1, arr2

    def analyse_flow(
        self, prev_field: Union[Frame, NDArray], current_field: Union[Frame, NDArray]
    ) -> NDArray:
        if isinstance(prev_field, Frame) and isinstance(current_field, Frame):
            prev_features = prev_field.get_feature_field()
            current_features = current_field.get_feature_field()
        elif isinstance(prev_field, np.ndarray) and isinstance(
            current_field, np.ndarray
        ):
            prev_features = prev_field
            current_features = current_field
        else:
            raise TypeError(
                "prev_field and current_field must both be of type Frame or NDArray"
            )

        # Check input fields are same shape
        prev_features, current_features = check_arrays(
            prev_features, current_features, equal_shape=True, ndim=2, dtype=int
        )

        # Determine a subdomain size if not provided
        if self.subdomain_size is None:
            self.subdomain_size = self.determine_sufficient_subdomain_size(
                prev_features.shape
            )

        # Check inputs, don't proceeed if not validated
        prev_features, current_features = self._check_inputs(
            prev_features, current_features
        )
        if not prev_features:
            return None, None

        # Get subdomains to calculate FFT over
        subdomain_shape = np.array(
            [self.subdomain_size, self.subdomain_size], dtype=int
        )
        subdomain_step = subdomain_shape / 2

        # Get tuple of indices of subdomains to iterate over
        # This will also check that the subdomain shape exactly fits the domain
        y_subdomain_bounds, x_subdomain_bounds = self.get_overlapping_subdomain_idxs(
            prev_features.shape, subdomain_shape
        )

        # Combine these idxs pairwise to define the subdomain bounds as a tuple
        # E.g., for subdomain size of 20 in y, produces [0, 10, 20, 30...]
        # and for subdomain size of 30 in x, produces [0, 15, 30, 45...]
        # this operation then produces ((0, 10), (10, 20), ...) for y
        # and ((0, 15), (15, 30)...) for x
        y_subdomain_bounds_tuple = itertools.pairwise(y_subdomain_bounds)
        x_subdomain_bounds_tuple = itertools.pairwise(x_subdomain_bounds)

        # Finally, get permutations of all xy subdomain bounds
        # E.g., for example above, produces ( ((0, 10), (0, 15)), ((0, 10), (15, 30)) )
        subdomain_bounds = itertools.product(
            y_subdomain_bounds_tuple, x_subdomain_bounds_tuple
        )

        # Initialise containing array for holding subdomain dy, dx
        # Shape is number of subdomains in each direction * 2 for overlaps
        containing_shape = (prev_features.shape // subdomain_shape) * 2
        subdomain_dy = np.full(shape=containing_shape, fill_value=np.nan)
        subdomain_dx = np.full(shape=containing_shape, fill_value=np.nan)

        for y_bounds, x_bounds in subdomain_bounds:
            # Construct subdomain mask from bounds
            y_slice = slice(y_bounds[0], y_bounds[1])
            x_slice = slice(x_bounds[0], x_bounds[1])
            subdomain_mask = (y_slice, x_slice)

            dy, dx = self.track_subdomain_flow(
                field1=prev_features[subdomain_mask],
                field2=current_features[subdomain_mask],
                tukey_filtering=self.apply_tukey_filtering,
            )

            # Use first bounds to get idx for dy and dx subdomain
            dy_idx = int(y_bounds[0] // subdomain_step[0])
            dx_idx = int(x_bounds[0] // subdomain_step[1])
            subdomain_dy[dy_idx, dx_idx] = dy
            subdomain_dx[dy_idx, dx_idx] = dx

        y_flow = self.interpolate_subdomain_flows(
            y_subdomain_bounds, x_subdomain_bounds, subdomain_dy, prev_features.shape
        )
        x_flow = self.interpolate_subdomain_flows(
            y_subdomain_bounds, x_subdomain_bounds, subdomain_dx, prev_features.shape
        )

        return y_flow, x_flow

    def determine_sufficient_subdomain_size(self, feature_field_shape):
        # TODO: figure out some logic here for getting a good sd size
        # if none is provided.
        # Use this for now, but it won't work in all cases!
        # TODO: this is entirely arbitrary. Check if this is sensible. It probably isnt
        # TODO: also need checks that this will divide the domain sensibly.
        # TODO: what if domain is an odd shape?? What then??
        return max(feature_field_shape[0], feature_field_shape[1]) // 5

    def check_subdomain_size_fits_in_full_domain(
        self, feature_field_shape: NDArray, subdomain_shape: NDArray
    ) -> bool:
        # Check that there an exact number of overlapping subdomains that can fit in the full field
        # First, check if subdomain shape/2 is an integer
        feature_field_shape, subdomain_shape = check_arrays(
            feature_field_shape, subdomain_shape, dtype=int, shape=(2,)
        )

        if not np.all(subdomain_shape % 2 == 0):
            return False

        subdomain_check = [
            dim % sd_shape / 2
            for dim, sd_shape in zip(feature_field_shape, subdomain_shape)
        ]
        if any([remainder != 0 for remainder in subdomain_check]):
            return False
        return True

    def get_overlapping_subdomain_idxs(
        self, feature_field_shape: NDArray, subdomain_shape: NDArray
    ) -> tuple[tuple]:
        """
        Get indices of subdomain bounds of the requested shape that will fit into
        the requested feature field. These subdomains overlap halfway, meaning that
        the requirement for an exact fit is that HALF the subdomain shape must fit.
        Returns tuples giving bounds as ((y0, y1), (x0, x1))

        Args:
            feature_field_shape (NDArray):
                Shape of the feature field to subdivide
            subdomain_shape (NDArray):
                Requested shape of the subdomain

        Raises:
            ValueError: If requested subdomain cannot fit exactly into input field

        Returns:
            tuple(tuple): overlapping subdomain bounds in form ((y0, y1), (x0, x1))
        """

        feature_field_shape, subdomain_shape = check_arrays(
            feature_field_shape, subdomain_shape, dtype=int, shape=(2,)
        )

        if not self.check_subdomain_size_fits_in_full_domain(
            feature_field_shape, subdomain_shape
        ):
            print(f"Input feature field dim size: {feature_field_shape}")
            print(f"Requested subdomain shape: {subdomain_shape}")
            msg = "Could not fit exact number of subdomains in feauture_field"
            raise ValueError(msg)

        # Now, get idxs of subdomain bounds to iterate over (with overlap)
        step = (subdomain_shape / 2).astype(int)
        y_subdomain_idxs = np.arange(
            start=0, stop=feature_field_shape[0] + step[0], step=step[0]
        )

        x_subdomain_idxs = np.arange(
            start=0, stop=feature_field_shape[1] + step[1], step=step[1]
        )

        return y_subdomain_idxs, x_subdomain_idxs

    def track_subdomain_flow(
        self, field1: NDArray, field2: NDArray, tukey_filtering: bool = True
    ) -> list[int]:
        """
        Uses FFT to identify most likely dy, dx motion vectors that translate field1
        to field 2. This is largely handled by skimage.registration.phase_cross_correlation
        but with additional pre-processing to avoid spurious correlations. E.g., if
        tukey_smoothing flag is enabled, applies a filter to the edges of each field
        that tapers to zero, which avoids spectral leakage.

        Args:
            field1 (NDArray):
                Previous timestep binary field
            field2 (NDArray):
                Current timestep binary field
            tukey_filtering (bool, optional):
                Whether to apply tukey filter to input fields to prevent wrap-around
                disparities occuring during FFT transformations.
                Defaults to True.

        Returns:
            list[int]: [dy, dx] motion vectors for subdomain flow
        """
        # Check inputs are equally shaped 2D arrays containing ints
        field1, field2 = check_arrays(
            field1, field2, ndim=2, equal_shape=True, dtype=int
        )

        # Filter inputs if flagged
        if tukey_filtering:
            domain_filter = self.get_2d_tukey_window(field1.shape)
            field1 = field1 * domain_filter
            field2 = field2 * domain_filter

        # Subtracting the mean from binary fields before cross correlation centres each
        # field around zero and improves accuracy of correlation peak
        m1 = field1 - np.mean(field1)
        m2 = field2 - np.mean(field2)

        # Since image registration finds the vector that translates the second arg to the
        # first, need to reverse input order
        cross_corr = phase_cross_correlation(
            m2,
            m1,
            space="real",
            overlap_ratio=self.overlap_threshold,
            normalization=None,
            upsample_factor=1,
            disambiguate=False,
        )

        dy, dx = cross_corr[0]
        error = cross_corr[1]
        return dy, dx

    def get_2d_tukey_window(self, arr_shape: NDArray) -> NDArray:
        """
        Creates a 2D tukey filter for an array of the requested input shape.
        Uses scipy.signal.windows.tukey to create a 1D filter for each dimension,
        then constructs the 2D filter using outer product of each 1D filter.

        A Tukey window is a tapered cosine combination of a rectangular and
        Hanning window which provides a good balance between preventing spectral
        leakage and maintaining good frequency resolution.

        Args:
            arr_shape (NDArray):
                Shape with which to create the Tukey filter


        Returns:
            NDArray: 2D filter array of requested shape
        """
        # Checks that the arr_shape input describes dimensions of a 2D array
        arr_shape = check_arrays(arr_shape, shape=(2,), dtype=int)

        # Create a 1D Tukey filter for each dimension. Alpha sets the degree to which
        # the filter resembles either a rectangular window (alpha=0) or a Hanning window
        # (alpha=1). We want to retain a lot of points if the subdomain is small, but for
        # larger subdomains we can apply more smoothing.
        filters = [
            tukey(dim_size, alpha=max(0.1, 10.0 / dim_size)) for dim_size in arr_shape
        ]

        return np.outer(*filters)

    def interpolate_subdomain_flows(
        self, y_subdomain_bounds, x_subdomain_bounds, subdomain_flows, full_domain_shape
    ) -> NDArray:
        # TODO: tidy and document this once it is working.
        # TODO: also need to include dd_tolerance here somehow??
        valid_mask = ~np.isnan(subdomain_flows)
        coords = np.array(np.nonzero(valid_mask)).T
        values = subdomain_flows[valid_mask]
        xmat, ymat = np.meshgrid(range(full_domain_shape[0]), full_domain_shape[1])
        if np.size(values) >= 4:
            it = LinearNDInterpolator(coords, values, fill_value=0)
            filled = it(list(np.ndindex(subdomain_flows.shape))).reshape(
                subdomain_flows.shape
            )

            # interp2d deprecated in newer version of scipy.
            # For functionally identical replacement, use RectBivariateSpline
            # with kx=3, ky=3 for cubic spline interpolation, and additional transposing.
            # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
            # fu = interpolate.interp2d(xint[0, :], yint[:, 0], filled, kind="cubic")
            fu = RectBivariateSpline(
                x_subdomain_bounds[0, :], y_subdomain_bounds[:, 0], filled.T, kx=3, ky=3
            )
            newumat = fu(xmat[0, :], ymat[:, 0]).T
        else:
            newumat = np.zeros(full_domain_shape)
        return newumat
