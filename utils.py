from numpy import dot

# gym enviroment params
BOARD_SZ = 4
MAP_NAME = "{}x{}".format(BOARD_SZ, BOARD_SZ)
TIME_LIMIT = 10*BOARD_SZ
FIXED_SEED = 2

# maps the flattened state displacement to the corresponding action
displacement_to_action = { 
     0 : -1,
    -BOARD_SZ : 3,
    -1 : 0,
     1 : 2,
     BOARD_SZ : 1 
}

# rewards and punishments coefficients
rp = {
    "completed" : 100.,
    "goal_direction" : 10.,
    "deliberate_action" : 5.,
    "lost_on_time" : -80.,
    "stuck" : -20.,
    "moving_backwards" : -20.,
    "fell_in_hole" : -200.,
}

def reward_engineering(state: int, prev_action: int, action: int, completed: int, next_state: int, terminated: bool, lost_on_time: bool) -> float:
    """
    Modifies rewards for better guided training in the Frozen Lake environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 1).
    :param prev_action: previous time step action.
    :type prev_action: int.
    :param action: action.
    :type action: int.
    :param completed: whether objective was reached
    :type completed: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 1).
    :param terminated: whether the episode ended with the action
    :type terminated: float.
    :param completed: whether objective was reached
    :type completed: float.
    :return: modified reward.
    :rtype: float.
    """
    displacement = next_state[0] - state[0]
    action_taken = displacement_to_action[displacement]
    deliberate_action = int(action_taken == action)
    goal_direction = 1 if action_taken == 2 or action_taken == 1 else -1
    stuck = int(displacement == 0)
    fell_in_hole = int(completed != 1 and terminated and not lost_on_time)
    moving_backwards = (prev_action - action) % 2
    loc = locals()
    states = list(map(loc.get, rp.keys()))
    return dot(list(rp.values()), states)