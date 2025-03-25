# features scrupt
import numpy as np

def extract_features_from_state_trajectory(state_array):
    """
    Given a trajectory (T x 22 state array), extract a set of features.
    Returns a 1D numpy array of features.
    """

    # Filtering out invalid rows (inf or NaN) 
    valid_mask = np.all(np.isfinite(state_array[:, [0,1,2, 7,8,9, 19,20,21]]), axis=1)
    state_array = state_array[valid_mask]
    if len(state_array) == 0:
        return np.zeros(6)  # fallback if all rows were invalid

    # Extracting the positions from state 
    ee_positions = state_array[:, 0:3]    # End-effector (robot hand)
    obj_positions = state_array[:, 7:10]  # Banana
    goal_positions = state_array[:, 19:22]  # Plate /goa
    obj_z = obj_positions[:, 2]  # z-position of banana

    #  ---features

    # Final distance from banana to goal
    final_obj = obj_positions[-1]
    final_goal = goal_positions[-1]
    final_dist_to_goal = np.linalg.norm(final_obj - final_goal)

    # the closest the banana got to the plate 
    min_dist_to_goal = np.min(np.linalg.norm(obj_positions - goal_positions, axis=1))

    #  Average distance from gripper to banana - small: hovers near . large :probably not anywhere clsoe most the time
    avg_grip_obj_dist = np.mean(np.linalg.norm(ee_positions - obj_positions, axis=1))

    #  Max height of the banana - kinda tells if it was ever picked up
    max_obj_z = np.max(obj_z)

    # Ratio of time banana was lifted above a threshold - kinda tells if it was ever actually held or maybe thrown /dropped 
    lift_threshold = 0.025
    held_ratio = np.sum(obj_z > lift_threshold) / len(obj_z)

    #  Binary success (if final distance is within 5cm)
    success = 1 if final_dist_to_goal < 0.05 else 0

    return np.array([
        final_dist_to_goal,
        min_dist_to_goal,
        avg_grip_obj_dist,
        max_obj_z,
        held_ratio,
        success
    ])