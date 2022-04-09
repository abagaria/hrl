def square_epsilon(subgoal, state):
    '''
    Labels state as being in the term set of the subgoal if it is within square epsilon of
    the subgoal position

    Args:
        subgoal (MonteRAMXYScreen || MonteRAMState): the ram state chosen to be subgoal
        state (MonteRAMState): the ram state to label

    Returns:
        (bool): whether state is in the term set of subgoal
    '''

    return abs(state.player_x - subgoal.player_x) <= 5 and abs(state.player_y - subgoal.player_y) <= 5

def square_epsilon_screen(subgoal, state):
    '''
    Labels state as being in the term set of the subgoal if it is within square epsilon of
    the subgoal position

    Args:
        subgoal (MonteRAMXYScreen || MonteRAMState): the ram state chosen to be subgoal
        state (MonteRAMState): the ram state to label

    Returns:
        (bool): whether state is in the term set of subgoal
    '''

    return abs(state.player_x - subgoal.player_x) <= 5 and abs(state.player_y - subgoal.player_y) <= 5 and state.screen == subgoal.screen
