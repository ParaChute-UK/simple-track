import datetime as dt

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from simpletrack.exceptions import IDError
from simpletrack.feature import Feature


def test_feature_init_with_1d_coords():
    time = dt.datetime.now()
    test_coords = np.array([0, 0])
    feature = Feature(1, test_coords, time)
    assert np.array_equal(feature.coords, test_coords)


def test_feature_init_with_2d_coords():
    time = dt.datetime.now()
    test_coords = np.array([[0, 1, 2], [0, 1, 2]])
    feature = Feature(1, test_coords, time)
    assert np.array_equal(feature.coords, test_coords)


def test_feature_init_with_3d_coord_raises_error():
    time = dt.datetime.now()
    test_coords = np.array([[[0, 0], [1, 1], [2, 2]]])
    with pytest.raises(ValueError):
        Feature(1, test_coords, time)


@pytest.fixture(scope="function")
def setup_test_feature():
    time = dt.datetime.now()
    test_coords = np.array([[0, 1, 2], [0, 1, 2]])
    feature = Feature(1, test_coords, time)
    return feature


def test_accreted_ids_added_correctly(setup_test_feature):
    test_feature = setup_test_feature

    # Add accreted ids to the feature
    test_feature.accrete_ids(10)
    test_feature.accrete_ids(11)

    # Check that these are correctly added to the accreted list
    assert test_feature.accreted == [10, 11]


def test_accreted_ids_correctly_replaced(setup_test_feature):
    test_feature = setup_test_feature

    # Add accreted ids to the feature
    test_feature.accrete_ids(10)
    test_feature.accrete_ids(11)

    # Now replace these with a new list of accreted ids
    test_feature.accrete_ids([12, 13], replace=True)

    # Check that the accreted list has been correctly replaced
    assert test_feature.accreted == [12, 13]


def test_accreted_ids_return_none_if_empty(setup_test_feature):
    test_feature = setup_test_feature

    # Check that the accreted list is None
    assert test_feature.accreted is None


def test_accreted_ids_return_none_if_replaced_with_empty(setup_test_feature):
    test_feature = setup_test_feature

    # Add accreted ids to the feature
    test_feature.accrete_ids(10)

    # Now replace these with an empty list
    test_feature.accrete_ids([], replace=True)

    # Check that the accreted list is now None
    assert test_feature.accreted is None


def test_spawned_ids_added_correctly(setup_test_feature):
    test_feature = setup_test_feature

    # Add accreted ids to the feature
    test_feature.spawns(10)
    test_feature.spawns(11)

    # Check that these are correctly added to the accreted list
    assert test_feature.children == [10, 11]


def test_spawned_ids_correctly_replaced(setup_test_feature):
    test_feature = setup_test_feature

    # Add accreted ids to the feature
    test_feature.spawns(10)
    test_feature.spawns(11)

    # Now replace these with a new list of accreted ids
    test_feature.spawns([12, 13], replace=True)

    # Check that the accreted list has been correctly replaced
    assert test_feature.children == [12, 13]


def test_spawned_ids_return_none_if_empty(setup_test_feature):
    test_feature = setup_test_feature

    # Check that the accreted list is None
    assert test_feature.children is None


def test_spawned_ids_return_none_if_replaced_with_empty(setup_test_feature):
    test_feature = setup_test_feature

    # Add accreted ids to the feature
    test_feature.spawns(10)

    # Now replace these with an empty list
    test_feature.spawns([], replace=True)

    # Check that the accreted list is now None
    assert test_feature.children is None


def test_is_new_with_no_further_initialisation(setup_test_feature):
    test_feature = setup_test_feature

    # Check that the feature is new
    assert test_feature.is_new() is True


def test_is_new_with_lifetime_and_parent_setter_and_reset(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.lifetime = 5
    # Reset to 1
    test_feature.lifetime = 1

    test_feature.parent = 10
    # Reset to None
    test_feature.parent = None

    # Check that the feature is new
    assert test_feature.is_new() is True


def test_is_new_with_lifetime_setter_only(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.lifetime = 5

    # Check that the feature is new
    assert test_feature.is_new() is False


def test_is_new_with_parent_id_setter_only(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.parent = 10

    # Check that the feature is new
    assert test_feature.is_new() is False


def test_is_new_with_lifetime_and_parent_setter(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.lifetime = 5
    test_feature.parent = 10

    # Check that the feature is new
    assert test_feature.is_new() is False


def test_is_dissipating_with_no_further_initialisation(setup_test_feature):
    test_feature = setup_test_feature

    # Check that the feature is not dissipating
    assert test_feature.is_dissipating() is False


def test_is_dissipating_with_final_timestep_and_accreted_setter(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.set_as_final_timestep()
    test_feature.accreted_in_next_frame_by = 10

    # Check that the feature is dissipating
    assert test_feature.is_dissipating() is False


def test_is_dissipating_with_accreted_setter(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.accreted_in_next_frame_by = 10

    # Check that the feature is dissipating
    assert test_feature.is_dissipating() is False


def test_is_dissipating_with_final_timestep_setter(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.set_as_final_timestep()

    # Check that the feature is dissipating
    assert test_feature.is_dissipating() is True


def test_is_final_timestep_with_no_further_initialisation(setup_test_feature):
    test_feature = setup_test_feature

    # Check that the feature is not in its final timestep
    assert test_feature.is_final_timestep() is False


def test_is_final_timestep_with_final_timestep_setter(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.set_as_final_timestep()

    # Check that the feature is in its final timestep
    assert test_feature.is_final_timestep() is True


# Functions that fill a space with a rectangle of arbitrary length, width, orientation, centre


def get_bounding_points_for_rect(length=50, width=4, orientation=0, centre=(50, 50)):
    # Get coords for rectangle corners
    half_length = length / 2
    half_width = width / 2
    corners = np.array(
        [
            [-half_length, -half_width],
            [-half_length, half_width],
            [half_length, half_width],
            [half_length, -half_width],
        ]
    )
    # Rotate these by the orientation angle
    rotation_matrix = np.array(
        [
            [np.cos(orientation), -np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)],
        ]
    )
    rotated_corners = corners @ rotation_matrix.T
    # Translate to the centre point
    translated_corners = rotated_corners + np.array(centre)
    return translated_corners.astype(int)


# Source - https://stackoverflow.com/a/42165596
# Posted by Charlie Brummitt
# Retrieved 2026-05-07, License - CC BY-SA 3.0


def point_in_hull(point, hull, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


def get_linear_feature_coords(
    length=50, width=4, orientation=0, centre=(50, 50), arr_shape=(500, 500)
):
    bounding_coords = get_bounding_points_for_rect(
        length=length, width=width, orientation=orientation, centre=centre
    )

    # Create ConvexHull that defines boundary parameters of shape given corner points
    hull = ConvexHull(bounding_coords)

    # # Not the fastest or most intelligent way to do this
    # # Iterate through all points in the arr_shape and get mask of points within hull
    # coord_mask = np.zeros(arr_shape, dtype=bool)
    # for y in range(arr_shape[0]):
    #     for x in range(arr_shape[1]):
    #         coord_mask[y, x] = point_in_hull([y, x], hull)

    # The below code is the vecotrised method of doing the above

    # Vectorized approach: create grid of all points
    y_coords, x_coords = np.meshgrid(
        np.arange(arr_shape[0]), np.arange(arr_shape[1]), indexing="ij"
    )
    points = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    # Check all points against all hull equations at once
    coord_mask = np.ones(arr_shape[0] * arr_shape[1], dtype=bool)
    for eq in hull.equations:
        coord_mask &= np.dot(points, eq[:-1]) + eq[-1] <= 1e-12

    # Reshape back to original array shape
    coord_mask = coord_mask.reshape(arr_shape)

    return coord_mask


def test_feature_elongation_calculation():
    test_data = np.zeros((500, 500))
    test_length = 100
    test_width = 50
    # Get mask containing ones within the rectangle of specified shape
    # Multiply length and width by 2 to turn radius -> diameter
    # NOTE: in future, this can just be done with skimage.draw module
    feature_coords = get_linear_feature_coords(
        length=test_length * 2,
        centre=(150, 250),
        orientation=np.pi / 6,  # 30 degrees
        width=test_width * 2,
        arr_shape=test_data.shape,
    )
    # Get (2,N) array of coords
    feature_coords = np.where(feature_coords == 1)
    (
        major_vector,
        minor_vector,
        major_radius,
        minor_radius,
    ) = Feature.calculate_major_minor_axes(None, feature_coords)

    major_axis_orientation = np.arctan2(major_vector[0], major_vector[1])
    minor_axis_orientation = np.arctan2(minor_vector[0], minor_vector[1])

    # Allow for 2 s.f. tolerance, it isn't perfect!
    np.testing.assert_approx_equal(np.pi / 6, major_axis_orientation, significant=2)
    np.testing.assert_approx_equal(4 * np.pi / 6, minor_axis_orientation, significant=2)

    np.testing.assert_approx_equal(test_length, major_radius, significant=2)
    np.testing.assert_approx_equal(test_width, minor_radius, significant=2)


def test_repr_method(setup_test_feature):
    test_feature = setup_test_feature
    repr_str = f"Feature id: {test_feature.id}, "
    repr_str += (
        f"lifetime: {test_feature.lifetime} timestep(s) at time: {test_feature.time}"
    )
    assert repr(test_feature) == repr_str


def test_accreted_in_next_frame_by_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.accreted_in_next_frame_by = 5
    assert test_feature.accreted_in_next_frame_by == 5


def test_major_vector_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature._major_vector = np.array([1, 0])
    assert np.array_equal(test_feature.major_vector, np.array([1, 0]))


def test_minor_vector_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature._minor_vector = np.array([0, 1])
    assert np.array_equal(test_feature.minor_vector, np.array([0, 1]))


def test_major_radius_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature._major_radius = 10
    assert test_feature.major_radius == 10


def test_minor_radius_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature._minor_radius = 5
    assert test_feature.minor_radius == 5


def test_coords_setter(setup_test_feature):
    test_feature = setup_test_feature
    new_coords = np.array([[1, 1], [2, 2], [3, 3]])
    test_feature.coords = new_coords
    assert np.array_equal(test_feature.coords, new_coords)
    assert test_feature.major_radius is not None
    assert test_feature.minor_radius is not None
    assert test_feature.major_vector is not None
    assert test_feature.minor_vector is not None


def test_coords_setter_single_coord(setup_test_feature):
    test_feature = setup_test_feature
    new_coords = np.array([1, 1])
    test_feature.coords = new_coords
    assert np.array_equal(test_feature.coords, new_coords)
    assert test_feature.major_radius is None
    assert test_feature.minor_radius is None
    assert test_feature.major_vector is None
    assert test_feature.minor_vector is None


def test_coords_setter_invalid_shape(setup_test_feature):
    test_feature = setup_test_feature
    new_coords = np.array([[[1, 1], [2, 2]]])  # Invalid shape (3D)
    with pytest.raises(ValueError):
        test_feature.coords = new_coords


def test_feature_equality(setup_test_feature):
    test_feature = setup_test_feature
    same_feature = Feature(test_feature.id, test_feature.coords, test_feature.time)
    different_feature = Feature(2, test_feature.coords, test_feature.time)

    assert test_feature == same_feature
    assert test_feature != different_feature


def test_provisional_id_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.provisional_id = 42
    assert test_feature.provisional_id == 42


def test_centroid_property_with_1d_coords():
    time = dt.datetime.now()
    test_coords = np.array([10, 20])
    feature = Feature(1, test_coords, time)
    assert feature.centroid == (10, 20)


def test_centroid_property_with_2d_coords():
    time = dt.datetime.now()
    test_coords = np.array([[10, 30, 50], [20, 40, 60]])
    feature = Feature(1, test_coords, time)
    expected_centroid = (30, 40)
    assert feature.centroid == expected_centroid


def test_parent_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.parent = 5
    assert test_feature.parent == 5


def test_dydx_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.dydx = (3, 4)
    assert test_feature.dydx == (3, 4)


def test_max_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.max = 100
    assert test_feature.max == 100


def test_mean_property(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.mean = 50
    assert test_feature.mean == 50


def test_id_setter(setup_test_feature):
    test_feature = setup_test_feature
    test_feature.id = 10
    assert test_feature.id == 10


def test_id_setter_invalid_type(setup_test_feature):
    test_feature = setup_test_feature
    with pytest.raises(IDError):
        test_feature.id = "invalid_id"  # Should raise IDError since id must be an int


def test_get_size(setup_test_feature):
    test_feature = setup_test_feature
    expected_size = 3
    assert test_feature.get_size() == expected_size


def test_get_size_with_1d_coords():
    time = dt.datetime.now()
    test_coords = np.array([10, 20])
    feature = Feature(1, test_coords, time)
    expected_size = 1
    assert feature.get_size() == expected_size


def test_summarise_headers_only(setup_test_feature):
    test_feature = setup_test_feature
    headers = test_feature.summarise(headers_only=True)
    expected_headers = [
        "id",
        "centroid",
        "size",
        "dydx",
        "max",
        "mean",
        "lifetime",
        "accreted",
        "parent",
        "children",
    ]
    assert headers == expected_headers


@pytest.mark.parametrize("output_type", ["str", "dict"])
def test_summarise_valid_output(setup_test_feature, output_type):
    test_feature = setup_test_feature
    summary = test_feature.summarise(headers_only=False, output_type=output_type)

    assert (
        isinstance(summary, str) if output_type == "str" else isinstance(summary, dict)
    )
    assert "id" in summary
    assert "centroid" in summary
    assert "size" in summary
    assert "dydx" in summary
    assert "max" in summary
    assert "mean" in summary
    assert "lifetime" in summary
    assert "accreted" in summary
    assert "parent" in summary
    assert "children" in summary


def test_summarise_invalid_output_type(setup_test_feature):
    test_feature = setup_test_feature
    with pytest.raises(ValueError):
        test_feature.summarise(headers_only=False, output_type="invalid_type")
