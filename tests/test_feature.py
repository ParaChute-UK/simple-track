import datetime as dt

import numpy as np
import pytest

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
