import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from Frame import Frame
from typing import Union
from utils import check_arrays
import itertools
from skimage.registration import phase_cross_correlation


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
        self.zero_pad_subdomains = True

    def _check_inputs(self, arr1: NDArray, arr2: NDArray) -> bool:
        arr1, arr2 = check_arrays(arr1, arr2, ndim=2, dtype=int)

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
        check_arrays(prev_features, current_features, equal_shape=True)

        # Determine a subdomain size if not provided
        if self.subdomain_size is None:
            self.subdomain_size = self.determine_sufficient_subdomain_size(
                prev_features.shape
            )

        # Check inputs, return None if inputs aren't validated
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
        # this produces ((0, 10), (10, 20), ...) for y
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
                zero_pad=self.zero_pad_subdomains,
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

    def get_overlapping_subdomains(
        self, feature_field: NDArray, subdomain_shape: NDArray
    ) -> NDArray:
        """
        Generate list/NDArray? of subdomains with overlapping boundaries

        Args:
            feature_field_shape (NDArray): _description_
            subdomain_shape (NDArray): _description_
        """

        # Check that there an exact number of subdomains that can fit in the full field
        subdomain_check = [
            dim % sd_shape
            for dim, sd_shape in zip(feature_field.shape, subdomain_shape)
        ]

        for dim, subdomain_remainder in enumerate(subdomain_check):
            if subdomain_remainder != 0:
                print(f"Input feature field dim size: {feature_field.shape[dim]}")
                print(f"Requested subdomain shape: {subdomain_shape[dim]}")
                msg = f"Could not fit exact number of subdomains in dimension {dim}"
                raise ValueError(msg)

        # TODO: add check for input types
        # TODO: add check for whether subdomain shape size is even for division by 2

        # Note there is overlap between the subdomains, hence divison by 2 in bounds
        stride = (subdomain_shape / 2).astype(int)

        # TODO: do I actually want the masks/different views, or do I want the bounds??
        # Use sliding_window to get subdomain masks
        # This function slides window by 1 idx, so use stride to select windows
        subdomains = sliding_window_view(feature_field, subdomain_shape)[
            ::stride, ::stride, ...
        ]

        return subdomains

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
        self, field1: NDArray, field2: NDArray, zero_pad: bool = True
    ):
        # TODO: if this becomes a func rather than a method, will need to add
        # some input checking. For now though, can assume inputs are okay
        # e.g., will need to check inputs are square, 2d, np arrays with
        # fields that are binary

        # TODO: will probably need scientific references or more justification
        # for the processes in this bit of code

        # TODO: need to make this dependent on the exact shape of the field, not
        # just use the max side length
        max_length = max(np.shape(field1))

        # TODO: make this "method" arg thing mean something
        # Seems to be whether to apply a sponge region to the hann1 array...
        # Note: this is actually a window function to reduce edge effects
        if zero_pad:
            # ???
            alpha = max(0.1, 10.0 / max_length)

            # midpoints between data, assuming square subdomains so can use the same midpoints
            # for x and y
            xhan = np.array(np.arange(0.5, max_length + 0.5))

            # Construct array of ones of same shape as midpoint array
            # This is a weights array? The following code changes weights at the start and end
            # so that weights go from 0 -> 1 -> 0 in a sponge region around the ones.
            # Weights are defined following a cos function.
            hann1 = np.ones_like(xhan)

            # Apply special consideration to upper and lower values
            # TODO: why does this define the sponge region limits?
            lower_limit = alpha * max_length / 2.0
            upper_limit = max_length - lower_limit

            lower_mask = xhan < lower_limit
            upper_mask = xhan > upper_limit

            # TODO: science reason for these sponge values and definitions.
            lower_sponge_vals = 0.5 * (
                1 + np.cos(np.pi * (2 * xhan[lower_mask] / (alpha * max_length) - 1))
            )

            # This appears to be just the lower_sponge_vals reversed: can this be simplified?
            upper_sponge_vals = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (2 * xhan[upper_mask] / (alpha * max_length) - 2.0 / alpha + 1)
                )
            )

            hann1[lower_mask] = lower_sponge_vals
            hann1[upper_mask] = upper_sponge_vals

        else:
            xhan = np.array(np.arange(0.5, max_length + 0.5))
            hann1 = np.ones([np.size(xhan)])

        # Then, take complex conjugate, transpose this, and multiply by hann1
        # TODO: For these 1D arrays, the conj and transpose doesn't do anything
        # So, hann2 is just hann1 squared... Is this correct?
        hann2 = hann1.conj().transpose() * hann1

        ## FIND CONVOLUTION S1, S2 USING FFT

        # b1, b2 are just the hann2 weights applied to field1, field2
        # Remember, field1 and field2 are binary
        # TODO: check if this is correct: field1 is 2D while hann2 is 1D!
        # So, hann2 is applied along all *rows* of elements, but not columns
        print(field1.shape)
        print(hann2.shape)
        b1 = field1 * hann2
        b2 = field2 * hann2

        # Standardise by the mean of the arrays, remembering field1 and field2 are binary
        # and likely filled with many zeros
        m1 = b1 - np.mean(b1)
        m2 = b2 - np.mean(b2)

        # get the fft wind. These end up being large values that need renormalising.
        # ffv = signal.fftconvolve(s1,s2,mode='same')

        # TODO: write explanation for these parameters and for *-1
        cross_corr = phase_cross_correlation(
            m1,
            m2,
            space="real",
            overlap_ratio=self.overlap_threshold,
            normalization=None,
            upsample_factor=1,
            disambiguate=False,
        )
        dy, dx = cross_corr[0] * -1
        return dy, dx

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
