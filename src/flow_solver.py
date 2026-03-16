import itertools
import warnings
from collections.abc import Iterable
from typing import Union

import numpy as np
import scipy.ndimage as ndimage
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy.signal.windows import tukey
from skimage.registration import phase_cross_correlation

from frame import Frame
from utils import ArrayError, check_arrays


class FlowSolver:
    def __init__(
        self,
        subdomain_size=None,
        min_fractional_coverage=0.01,
        subdomain_tolerance=3,
        overlap_threshold=0.6,
        apply_tukey_filtering=True,
    ) -> None:
        if isinstance(subdomain_size, int):
            self.subdomain_shape = np.array([subdomain_size, subdomain_size], dtype=int)
        elif isinstance(subdomain_size, float):
            raise TypeError("Expected int or array-like, got float")
        elif subdomain_size is None:
            self.subdomain_shape = None
        else:
            self.subdomain_shape = check_arrays(subdomain_size, ndim=2).astype(int)
        self.min_fractional_coverage = min_fractional_coverage
        self.subdomain_tolerance = subdomain_tolerance
        self.overlap_threshold = overlap_threshold
        self.apply_tukey_filtering = apply_tukey_filtering

    def analyse_flow(
        self, prev_field: Union[Frame, NDArray], current_field: Union[Frame, NDArray]
    ) -> list[NDArray, NDArray]:
        """
        Analyses previous field and current field to identify flow field. Uses phase
        cross correlation over a series of overlapping subdomains to stitch together
        a full field, where the flow is constant within each subdomain. Subdomain size
        is controlled by OpticalFlowSolver init, but is estimated from inputs if not
        provided.

        Input feature fields must be of the same size and contain sufficient feature
        coverage, as determined by min_fractional_coverage in init. Otherwise, solver
        will return None for the flow fields.

        Args:
            prev_field (Union[Frame, NDArray]):
                Feature field from previous timestep
            current_field (Union[Frame, NDArray]):
                Feature field from current timestep

        Returns:
            list[NDArray, NDArray]: y_flow, x_flow
        """
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
        if self.subdomain_shape is None:
            self.subdomain_shape = self.get_subdomain_shape(prev_features.shape)

        # Check inputs, don't proceed if not validated
        prev_features, current_features = self._check_inputs(
            prev_features, current_features
        )
        if prev_features is None:
            return None, None

        # Initialise containing arrays for holding subdomain dy, dx
        subdomain_dy, subdomain_dx = self.get_subdomain_containment_arrays(
            prev_features.shape, self.subdomain_shape
        )

        # Get tuple of indices of subdomains to iterate over
        # This will also check that the subdomain shape exactly fits the domain
        y_subdomain_bounds, x_subdomain_bounds = self.get_overlapping_subdomain_idxs(
            prev_features.shape, self.subdomain_shape
        )
        # Get the iterable of subdomain bounds
        subdomain_bounds = self.subdomain_iter(y_subdomain_bounds, x_subdomain_bounds)

        for y_bounds, x_bounds in subdomain_bounds:
            # Construct subdomain mask from bounds
            y_slice = slice(y_bounds[0], y_bounds[1])
            x_slice = slice(x_bounds[0], x_bounds[1])
            subdomain_mask = (y_slice, x_slice)

            dy, dx = self.derive_subdomain_flow(
                field1=prev_features[subdomain_mask],
                field2=current_features[subdomain_mask],
                tukey_filtering=self.apply_tukey_filtering,
            )

            # Use first bounds to get idx for dy and dx subdomain
            subdomain_step = self.subdomain_shape / 2
            dy_idx = int(y_bounds[0] // subdomain_step[0])
            dx_idx = int(x_bounds[0] // subdomain_step[1])
            subdomain_dy[dy_idx, dx_idx] = dy
            subdomain_dx[dy_idx, dx_idx] = dx

        # Check neighbouring subdomain values vary within acceptable tolerance
        subdomain_dy = self.check_subdomain_variability(subdomain_dy)
        subdomain_dx = self.check_subdomain_variability(subdomain_dx)

        # Finally, interpolate values between subdomains.
        # For this function, only need the interior subdomain bounds (not edge indices)
        interior_y_subdom_bounds = y_subdomain_bounds[1:-1]
        interior_x_subdom_bounds = x_subdomain_bounds[1:-1]
        y_flow = self.interpolate_subdomain_flows(
            interior_y_subdom_bounds,
            interior_x_subdom_bounds,
            subdomain_dy,
            prev_features.shape,
        )
        x_flow = self.interpolate_subdomain_flows(
            interior_y_subdom_bounds,
            interior_x_subdom_bounds,
            subdomain_dx,
            prev_features.shape,
        )
        return y_flow, x_flow

    def get_subdomain_containment_arrays(
        self, full_domain_shape: NDArray, subdomain_shape: NDArray
    ) -> NDArray:
        """
        Return array with correct shape for containing subdomain flow values
        Shape is number of subdomains in each direction * 2 for overlaps
        E.g., for a 100x200 domain and a 20x20 subdomain size, shape of
        containing arrays will be 10x20, but -1 from each dimension
        due to the stride of the values

        Args:
            full_domain_shape (NDArray): Shape of the full domain
            subdomain_shape (NDArray): Shape of requested subdomain

        Returns:
            NDArray: Array of NaNs of the required shape for containing subdomain data
        """
        full_domain_shape, subdomain_shape = check_arrays(
            full_domain_shape, subdomain_shape, shape=(2,), dtype=int, non_negative=True
        )

        containing_shape = (full_domain_shape // subdomain_shape) * 2
        containing_shape = [dim - 1 for dim in containing_shape]
        subdomain_dy = np.full(shape=containing_shape, fill_value=np.nan)
        subdomain_dx = np.full(shape=containing_shape, fill_value=np.nan)
        return subdomain_dy, subdomain_dx

    def subdomain_iter(
        self, y_subdomain_bounds: tuple, x_subdomain_bounds: tuple
    ) -> Iterable:
        """
        Produces iterable of subdomain bounds with stride of 2 indices
        between inputs. Each returned set is an iterable of tuples defining
        start and end bounds in y and x direction

        Args:
            y_subdomain_bounds (tuple):
            x_subdomain_bounds (tuple):

        Returns:
            Iterable: ((y_start, y_stop), (x_start, x_stop))
        """
        y_subdomain_bounds, x_subdomain_bounds = check_arrays(
            y_subdomain_bounds, x_subdomain_bounds, dtype=int, ndim=1, non_negative=True
        )

        # Combine these idxs pairwise with stride 2 to define the subdomain bounds
        # E.g., for subdomain size of 20 in y with bounds [0, 10, 20, 30...]
        # and for subdomain size of 30 in x with bounds [0, 15, 30, 45...]
        # this operation then produces ((0, 20), (10, 30), ...) for y
        # and ((0, 30), (15, 45)...) for x
        y_subdomain_bounds_tuple = pairwise_with_stride(y_subdomain_bounds, 2)
        x_subdomain_bounds_tuple = pairwise_with_stride(x_subdomain_bounds, 2)

        # Finally, get permutations of all xy subdomain bounds
        # E.g., for example above, produces ( ((0, 20), (0, 30)), ((0, 20), (15, 45))...)
        subdomain_bounds = itertools.product(
            y_subdomain_bounds_tuple, x_subdomain_bounds_tuple
        )
        return subdomain_bounds

    def check_subdomain_variability(self, subdomain_vals: NDArray):
        """
        Check variability in neighbouring subdomains to ensure no large
        discrepancies. If there is a neighbourhood mean that departs from
        the local value by more than self.subdomain_tolerance, set the
        local value to np.nan


        Args:
            subdomain_vals (NDArray):
                Flow values derived in subdomains

        Returns:
            NDArray: Input values with outliers replaced by nan
        """
        # Check input array is 2D
        subdomain_vals = check_arrays(subdomain_vals, ndim=2, dtype=float)

        # Setup footprint for performing filter check on neighbouring points
        # footprint excludes current index
        footprint = np.ones((3, 3))
        footprint[1, 1] = 0

        # Catch OOB warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nbhood_mean = ndimage.generic_filter(
                input=subdomain_vals,
                function=np.nanmean,  # Apply nanmean to each nbhood
                footprint=footprint,  # Deterrmines how to sample points in the nbhood
                mode="constant",  # Determines how to handle boundaries. "constant" = fill with cval
                cval=np.nan,  # Fill boundary values with nan so they don't contribute to nanmean
            )

        # Check for any values where the nanmean exceeds threshold set in init
        invalid_tolerance = (
            np.abs(nbhood_mean - subdomain_vals) > self.subdomain_tolerance
        )
        subdomain_vals[invalid_tolerance] = np.nan
        return subdomain_vals

    def get_subdomain_shape(self, feature_field_shape):
        # TODO: figure out some logic here for getting a good sd size
        # if none is provided.
        # Use this for now, but it won't work in all cases!
        # TODO: this is entirely arbitrary. Check if this is sensible. It probably isnt
        # TODO: what if domain is an odd shape?? What then??
        sd_shape = np.array(feature_field_shape) // 5
        if not self.check_subdomain_size_fits_in_full_domain(
            feature_field_shape, sd_shape
        ):
            # TODO: do something more intelligent here rather than just raise an error
            # Try to find another subdomain shape that could fit
            raise Exception(
                f"Derived subdomain shape ({sd_shape}) cannot fit ({feature_field_shape})"
            )
        return sd_shape

    def check_subdomain_size_fits_in_full_domain(
        self, feature_field_shape: NDArray, subdomain_shape: NDArray
    ) -> bool:
        """
        Determines whether the subdomain shape is suitable for the feature field
        shape. Subdomain is suitable only if it fits an equal number of times
        into the feature field in each dimension

        Args:
            feature_field_shape (NDArray):
                1D array describing shape of feature field
            subdomain_shape (NDArray):
                1d array describing shape of requested subdomain

        Returns:
            bool: True if subdomain shape fits exactly in feature field shape, False otherwise
        """
        # Check inputs, only except errors related to contents of inputs
        try:
            feature_field_shape, subdomain_shape = check_arrays(
                feature_field_shape,
                subdomain_shape,
                dtype=int,
                shape=(2,),
                non_negative=True,
            )
        except ArrayError:
            return False

        # First, check if subdomain shape/2 is an integer
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
            feature_field_shape,
            subdomain_shape,
            dtype=int,
            shape=(2,),
            non_negative=True,
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

    def derive_subdomain_flow(
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
            field1, field2, ndim=2, equal_shape=True, dtype=int, non_negative=True
        )
        if not isinstance(tukey_filtering, bool):
            raise TypeError(
                f"Expected tukey_filtering type bool, got {type(tukey_filtering)}"
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
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
        # error = cross_corr[1]
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
        arr_shape = check_arrays(arr_shape, shape=(2,), dtype=int, non_negative=True)

        # Create a 1D Tukey filter for each dimension. Alpha sets the degree to which
        # the filter resembles either a rectangular window (alpha=0) or a Hanning window
        # (alpha=1). We want to retain a lot of points if the subdomain is small, but for
        # larger subdomains we can apply more smoothing.
        filters = [
            tukey(dim_size, alpha=max(0.1, 10.0 / dim_size)) for dim_size in arr_shape
        ]

        return np.outer(*filters)

    def interpolate_subdomain_flows(
        self,
        y_subdomain_bounds: tuple,
        x_subdomain_bounds: tuple,
        subdomain_flows: NDArray,
        full_domain_shape: NDArray,
    ) -> NDArray:
        """
        Takes the subdomain flows found from track_subdomain_flow and stitches them
        together using 2d interpolation via RectBivariateSpline

        Args:
            y_subdomain_bounds (tuple):
                bounds of y subdomains
            x_subdomain_bounds (tuple):
                bounds of x subdomains
            subdomain_flows (NDArray):
                subdomain flows
            full_domain_shape (NDArray):
                Shape of full domain to create flow field for

        Returns:
            NDArray: Interpolated flow field for the full domain
        """
        # Check inputs
        y_subdomain_bounds, x_subdomain_bounds = check_arrays(
            y_subdomain_bounds, x_subdomain_bounds, ndim=1, non_negative=True
        )
        full_domain_shape = check_arrays(full_domain_shape, shape=(2,), dtype=int)
        subdomain_flows = check_arrays(subdomain_flows, ndim=2)
        subdomain_flows = self._fill_nans(subdomain_flows)

        # If there are fewer than 4 nonzero subdomain flow elements, return 0 flow field
        if np.count_nonzero(subdomain_flows) < 4:
            return np.zeros(full_domain_shape, dtype=int)

        # interp2d deprecated in newer version of scipy.
        # For functionally identical replacement, use RectBivariateSpline
        # with kx=3, ky=3 for cubic spline interpolation
        # RectBivariateSpline expected data on (x,y) grid, not expected (y,x),
        # hence transposed input and output.
        # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
        # fu = interpolate.interp2d(xint[0, :], yint[:, 0], filled, kind="cubic")
        fu = RectBivariateSpline(
            x_subdomain_bounds, y_subdomain_bounds, subdomain_flows.T, kx=3, ky=3
        )
        y_range = range(full_domain_shape[0])
        x_range = range(full_domain_shape[1])
        newumat = fu(x_range, y_range).T
        return newumat

    def _check_inputs(self, arr1: NDArray, arr2: NDArray) -> bool:
        # Check both fields have features
        if not np.count_nonzero(arr1) and not np.count_nonzero(arr2):
            print("No features detected in both fields. Skipping optical flow.")
            return None, None

        # If there are too few features, don't proceed with optical flow
        # TODO: what is actually the check here??
        subdomain_count = np.prod(self.subdomain_shape)
        min_feature_coverage = subdomain_count * self.min_fractional_coverage
        if np.sum(arr1) < min_feature_coverage or np.sum(arr2) < min_feature_coverage:
            print(f"Threshold for running optical flow: {self.min_fractional_coverage}")
            print(f"Number of pixels above threshold in arr1: {np.sum(arr1)}")
            print(f"Number of pixels above threshold in arr2: {np.sum(arr2)}")
            print("Number of features in arr1 and/or arr2 less than threshold. ")
            print("Skipping optical flow")
            return None, None

        return arr1, arr2

    def _fill_nans(self, arr: NDArray) -> NDArray:
        """
        Replace NaNs in the input field with values interpolated from neighbouring
        grid points, or 0 if this is not possible.

        Args:
            arr (NDArray): Input array, potentially containing NaNs
        Returns:
            NDArray: Ouput array with NaNs filled
        """
        # Create NaN mask
        valid_mask = ~np.isnan(arr)
        coords = np.nonzero(valid_mask)
        non_nan_values = arr[valid_mask]
        it = LinearNDInterpolator(coords, non_nan_values, fill_value=0)
        filled = it(list(np.ndindex(arr.shape))).reshape(arr.shape)
        return filled


def pairwise_with_stride(input_iter: Iterable, stride: int):
    """
    Similar to itertools.pairwise but with step between elements

    pairwise_with_stride('ABCDEFG', 1) → AB BC CD DE EF FG
    pairwise_with_stride('ABCDEFG', 2) → AC BD CE DF EG
    """

    if not isinstance(stride, int):
        raise TypeError(f"Expected int, got f{type(stride)}")

    pairwise_list = []
    for idx, element in enumerate(input_iter):
        try:
            next_element = input_iter[idx + stride]
            pairwise_list.append((element, next_element))
        except IndexError:
            pass

    return iter(pairwise_list)
