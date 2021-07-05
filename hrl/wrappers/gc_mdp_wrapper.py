from .mdp_wrapper import MDPWrapper

class GoadConditionedMDPWrapper(MDPWrapper):
    """
    this wrapper represents a goal conditioned MDP
    """
    def __init__(self, env) -> None:
        super().__init__()