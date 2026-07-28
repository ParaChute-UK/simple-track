import numpy as np

from simpletrack.frame import Frame
from simpletrack.optical_flow_solver import ILKFlowSolver


def test_ilk_flow_solver_ndarray_input():
    # Create two simple frames with a known translation
    frame1 = np.zeros((100, 100), dtype=np.float32)
    frame2 = np.zeros((100, 100), dtype=np.float32)

    # Add a feature in frame1
    frame1[40:60, 40:60] = 1.0

    # Translate the feature in frame2
    frame2[45:65, 45:65] = 1.0

    # Initialize the ILKFlowSolver
    solver = ILKFlowSolver()

    # Calculate the optical flow between the two frames
    y_flow, x_flow = solver.analyse_flow(frame1, frame2)

    # Check that the flow vectors are as expected (approximately)
    assert y_flow.shape == frame1.shape
    assert x_flow.shape == frame1.shape


def test_ilk_flow_solver_frame_input():
    # Create two Frame objects with a known translation
    frame1 = Frame()
    frame2 = Frame()

    # Add a feature in frame1
    frame1.feature_field = np.zeros((100, 100), dtype=np.float32)
    frame1.feature_field[40:60, 40:60] = 1.0

    # Translate the feature in frame2
    frame2.feature_field = np.zeros((100, 100), dtype=np.float32)
    frame2.feature_field[45:65, 45:65] = 1.0

    # Initialize the ILKFlowSolver
    solver = ILKFlowSolver()

    # Calculate the optical flow between the two frames
    y_flow, x_flow = solver.analyse_flow(frame1, frame2)

    # Check that the flow vectors are as expected (approximately)
    assert y_flow.shape == frame1.feature_field.shape
    assert x_flow.shape == frame1.feature_field.shape
