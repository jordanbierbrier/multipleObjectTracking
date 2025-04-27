import numpy as np
from scipy.optimize import linear_sum_assignment
from associate import calculate_iou

def assign(states, detections):
    """
    Assign detections to states using the Hungarian algorithm.

    Parameters:
        states (list): List of current states.
        detections (list): List of new detections.

    Returns:
        tuple: Indices of assigned states and detections, and their associated costs.
    """
    n_states = len(states)
    n_detections = len(detections)

    # Determine the size of the cost matrix
    size = max(n_states, n_detections)

    # Initialize the cost matrix with zeros
    cost_matrix = np.zeros((size, size))

    # Populate the cost matrix with IoU-based costs
    for i in range(n_states):
        for j in range(n_detections):
            cost_matrix[i][j] = 1 - calculate_iou(states[i], detections[j], xywh_rep=True)

    # Solve the assignment problem using the Hungarian algorithm
    state_indices, detection_indices = linear_sum_assignment(cost_matrix)

    # Extract the costs for the assigned pairs
    assignment_costs = np.zeros(len(state_indices))
    for i in range(len(state_indices)):
        assignment_costs[i] = cost_matrix[state_indices[i]][detection_indices[i]]

    return state_indices, detection_indices, assignment_costs
