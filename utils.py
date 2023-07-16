BOARD_SZ = 4
MAP_NAME = "{}x{}".format(BOARD_SZ, BOARD_SZ)
ACTION_N = 4
TIME_LIMIT = 10*BOARD_SZ
FIXED_SEED = 2

displacement_to_action = { 
     0 : -1,
    -BOARD_SZ : 3,
    -1 : 0,
     1 : 2,
     BOARD_SZ : 1 
}

def reward_engineering(state, prev_action, action, completed, next_state, terminated, time_over):
    """
    Makes reward engineering to better guided training in the Frozen Lake environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 1).
    :param action: action.
    :type action: int.
    :param reward: original reward.
    :type reward: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 1).
    :return: modified reward.
    :rtype: float.
    """
    a = 10.
    b = 2.
    d = next_state[0] - state[0]
    action_taken = displacement_to_action[d]
    #print(d)
    #print(action_taken)
    correct_direc = 1 if action_taken == 2 or action_taken == 1 else -1
    got_stuck = (d == 0)
    fell_in_a_hole = completed != 1 and terminated and not time_over
    regressing = (prev_action - action) % 2 
    return 100*completed-100*fell_in_a_hole-20*(got_stuck+regressing)-80*time_over+a*correct_direc+b*(action_taken == action)

