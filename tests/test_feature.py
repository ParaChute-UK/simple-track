import datetime as dt

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from simpletrack.feature import Feature


@pytest.fixture(scope="function")
def setup_test_feature():
    time = dt.datetime.now()
    test_coords = np.array([[0, 0], [1, 1], [2, 2]])
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
