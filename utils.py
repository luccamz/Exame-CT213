from numpy import dot

# gym enviroment params
BOARD_SZ = 4
BOARD_SHAPE = (BOARD_SZ, BOARD_SZ)
MAP_NAME = "{}x{}".format(BOARD_SZ, BOARD_SZ)
TIME_LIMIT = 25*BOARD_SZ
FIXED_SEED = 0
SLIPPERY = True

# maps the flattened state displacement to the corresponding action
displacement_to_action = { 
     0 : 6,
    -BOARD_SZ : 3,
    -1 : 0,
     1 : 2,
     BOARD_SZ : 1 
}

# rewards and punishments coefficients
rp = {
    "completed" : 100.,
    "goal_direction" : 1.,
    #"deliberate_action" : 2.,
    #"lost_on_time" : -80.,
    "stuck" : -3.,
    #"moving_backwards" : -10.,
    "fell_in_hole" : -1000.,
    #"at_corner" : -100
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
    displacement = next_state - state
    action_taken = displacement_to_action[displacement]
    deliberate_action = int(action_taken == action)
    goal_direction = 2 + 0.5*(action_taken % 3 != 0) if (action % 3) != 0 else -1
    stuck = int(displacement == 0)
    fell_in_hole = int(completed == 0 and terminated and not lost_on_time)
    moving_backwards = (1 + prev_action - action) % 2
    at_corner = 1 if state == 12 or state == 3 else 0
    loc = locals()
    states = list(map(loc.get, rp.keys()))
    return dot(list(rp.values()), states) 

action_dir = ['L','D','R','U']
act_arrows = ["←","↓","→","↑"]