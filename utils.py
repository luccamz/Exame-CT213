from numpy import dot
import json

# gym enviroment params

# user defined params (set these before training and don't change before evaluation)
BOARD_SZ = 6 # side length for the frozen lake env board
SLIPPERY = True # whether ground is slippery or not

# dependent params (don't change these)
BOARD_SHAPE = (BOARD_SZ, BOARD_SZ) 
MAP_NAME = "{}x{}".format(BOARD_SZ, BOARD_SZ)
TIME_LIMIT = 25*BOARD_SZ # number of time steps before truncation
FIXED_SEED = 0 # seed for random map generation

# maps the flattened state displacement to the corresponding action
displacement_to_action = { 
     0 : 6,
    -BOARD_SZ : 3,
    -1 : 0,
     1 : 2,
     BOARD_SZ : 1 
}

#if True, then loads weigths for general purpose, else, loads optimized weigths for particular problem
# that being the 6 x 6 slippery with seed 0
GENERAL_RP = True # TODO ! User, don't change this to False

# rewards and punishments coefficients
path = 'general.json' if GENERAL_RP else 'particular.json'

with open(path, 'r') as f:
    rp = json.load(f)

def reward_engineering(state: int, prev_action: int, action: int, completed: int, next_state: int, terminated: bool, lost_on_time: bool) -> float:
    """
    Modifies rewards for better guided training in the Frozen Lake environment.

    :param state: state.
    :type state: int.
    :param prev_action: previous time step action.
    :type prev_action: int.
    :param action: current time step action.
    :type action: int.
    :param completed: whether objective was reached (1 if reached, 0 otherwise).
    :type completed: int.
    :param next_state: next state.
    :type next_state: int.
    :param terminated: whether the episode ended with the action
    :type terminated: bool.
    :param lost_on_time: whether truncation happened (reached time limit)
    :type lost_on_time: bool.
    :return: modified reward.
    :rtype: float.
    """

    # the way this works is, you add an entry to the json file containing the weight, then
    # in the body of this function you define a variable with the exact same name, that should evaluate
    # to 0 if the weight shouldn't be applied and to 1 if it should (so, essentially a boolean). 
    # There's no need to consider signs in here because those go in the weights. If you want to exclude any
    # weight then you can either delete its entry from the json from which it's being read or delete the local 
    # variable here (or set either one to 0, obviously), therefore, it's a better practice to control it 
    # thorugh the json file, so you don't need to change the function each time
     
    displacement = next_state - state
    action_taken = displacement_to_action[displacement]
    deliberate_action = int(action_taken == action)
    goal_direction = (action % 3 != 0)
    went_goal_direction =  (action_taken % 3 != 0)
    opposite_direction = (action % 3 == 0)
    stuck = int(displacement == 0)
    fell_in_hole = int(completed == 0 and terminated and not lost_on_time)
    towards_hole = int(fell_in_hole and deliberate_action)
    moving_backwards = (1 + prev_action - action) % 2
    loc = locals()
    states = list(map(loc.get, rp.keys()))
    return dot(list(rp.values()), states) 

# maps (index -> value) action numbers to their direction initials
action_dir = ['L','D','R','U']

# maps action numbers to arrows in their respective direcion
act_arrows = ["←","↓","→","↑"]