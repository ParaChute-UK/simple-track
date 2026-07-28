import numpy as np
import pytest

from simpletrack.frame import Frame
from simpletrack.optical_flow_solver import DISFlowSolver, normalize8


def test_normalize8():
    # Test with a simple 2D array
    arr = np.array([[0, 1], [2, 3]], dtype=np.float32)
    normalized = normalize8(arr)
    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255

    # Test with a larger range of values
    arr = np.array([[10, 20], [30, 40]], dtype=np.float32)
    normalized = normalize8(arr)
    assert normalized.min() == 0
    assert normalized.max() == 255

    # Test with negative values
    arr = np.array([[-10, -5], [0, 5]], dtype=np.float32)
    normalized = normalize8(arr)
    assert normalized.min() == 0
    assert normalized.max() == 255
    assert normalized.dtype == np.uint8


@pytest.mark.parametrize(
    "field_shape, expected_patch_size",
    [
        ((100, 100), 20),  # Square field
        ((50, 200), 10),  # Rectangular field
        ((10, 10), 5),  # Small field (minimum patch size)
    ],
)
def test_get_patch_size(field_shape, expected_patch_size):
    dis_solver = DISFlowSolver()

    patch_size = dis_solver.get_patch_size(field_shape)
    assert patch_size == expected_patch_size


def test_analyse_flow_ndarray_input():
    dis_solver = DISFlowSolver()

    # Create dummy previous and current features
    prev_features = np.zeros((100, 100))
    current_features = np.zeros((100, 100))
    prev_features[30:70, 30:70] = 1  # Add a square feature
    current_features[32:72, 32:72] = 1  # Move the feature slightly

    # Analyse flow
    y_flow, x_flow = dis_solver.analyse_flow(prev_features, current_features)

    # Check that the output shapes match the input shapes
    assert y_flow.shape == prev_features.shape
    assert x_flow.shape == prev_features.shape


def test_analyse_flow_frame_input():
    dis_solver = DISFlowSolver()

    # Create dummy previous and current features
    prev_features = np.zeros((100, 100))
    current_features = np.zeros((100, 100))
    prev_features[30:70, 30:70] = 1  # Add a square feature
    current_features[32:72, 32:72] = 1  # Move the feature slightly

    prev_frame = Frame()
    current_frame = Frame()
    prev_frame.feature_field = prev_features
    current_frame.feature_field = current_features

    # Analyse flow
    y_flow, x_flow = dis_solver.analyse_flow(prev_frame, current_frame)

    # Check that the output shapes match the input shapes
    assert y_flow.shape == prev_features.shape
    assert x_flow.shape == prev_features.shape


def test_analyse_flow_invalid_input():
    dis_solver = DISFlowSolver()

    # Create invalid input (not a Frame or ndarray)
    invalid_input = "invalid"

    with pytest.raises(TypeError):
        y_flow, x_flow = dis_solver.analyse_flow(invalid_input, invalid_input)


def test_analyse_flow_empty_input():
    dis_solver = DISFlowSolver()

    # Create empty input arrays
    prev_features = np.zeros((100, 100))
    current_features = np.zeros((100, 100))

    # Analyse flow
    y_flow, x_flow = dis_solver.analyse_flow(prev_features, current_features)

    # Check that the output is None for both flows
    assert y_flow is None
    assert x_flow is None
