from gym import Wrapper

class MDPWrapper(Wrapper):
    """
    a wrapper to keep MDP functionalities
    """
    def __init__(self, env) -> None:
        super().__init__()