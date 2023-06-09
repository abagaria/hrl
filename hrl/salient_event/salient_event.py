import random
import numpy as np
from collections import deque
from pfrl.wrappers import atari_wrappers
from hrl.agent.dsc.utils import info_to_pos
from hrl.agent.dsc.datastructures import TrainingExample


class SalientEvent:
    def __init__(self, target_obs, target_info, tol):
        assert isinstance(tol, float)
        assert isinstance(target_info, dict)
        assert isinstance(target_obs, atari_wrappers.LazyFrames)

        self.tolerance = tol
        self.target_obs = target_obs
        self.target_info = target_info
        self.effect_set = deque([TrainingExample(target_obs, target_info)], maxlen=20)

        # Logging
        self.n_expansion_attempts = 0
        self.n_expansions_completed = 0

    @property
    def target_pos(self):
        return info_to_pos(self.target_info)

    def add_to_effect_set(self, obs, info):
        self.effect_set.append(TrainingExample(obs, info))

    def get_target_position(self):
        return self.target_info

    def get_target_obs(self):
        return self.target_obs

    def sample(self):
        """ Sample a state from the effect set. """ 
        if len(self.effect_set) > 0:
            sampled_point = random.choice(self.effect_set)
            return sampled_point.obs, sampled_point.info
        return self.target_obs, self.target_info

    def __call__(self, info):
        if isinstance(info, dict):
            pos = info['player_x'], info['player_y']
        else:
            import ipdb; ipdb.set_trace()
            
        xcond = abs(pos[0] - self.target_info["player_x"]) <= self.tolerance
        ycond = abs(pos[1] - self.target_info["player_y"]) <= self.tolerance
        room_cond = info["room_number"] == self.target_info["room_number"]

        inventory_cond = info["inventory"] == self.target_info["inventory"]
        left_door_cond = info["left_door_open"] == self.target_info["left_door_open"]
        right_door_cond = info["right_door_open"] == self.target_info["right_door_open"]

        return xcond and ycond and inventory_cond and room_cond and left_door_cond and right_door_cond

    def __str__(self):
        info = self.target_info
        if info["room_number"] == 1 and info["inventory"] == "00000000":
            return f"SE({self.target_pos})"
        return f"SE({self.target_info})"
    
    def __repr__(self):
        return str(self)

    def distance(self, info):
        """ Distance from current salient event to the input salient event. """
        pos = info_to_pos(info)

        if self.target_info["room_number"] == info["room_number"]:
            x_dist = abs(pos[0] - self.target_info["player_x"])
            y_dist = abs(pos[1] - self.target_info["player_y"])
            return np.sqrt(x_dist**2 + y_dist**2)
        return np.inf
