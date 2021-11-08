import cv2
import random
import numpy as np
from hrl.tasks.monte.MRRAMStateClass import MRRAMState
from hrl.wrappers.gym_wrappers import make_atari, wrap_deepmind

# TODO support mdp.sample_goal()
#              mdp.reward_function(state, goal)
class MontezumaRAMMDP:
    def __init__(self, render, seed):
        self.env_name = "MontezumaRevengeNoFrameskip-v4"
        self.env = wrap_deepmind(
            make_atari(env_id=self.env_name), 
            episode_life=True, 
            frame_stack=True,)

        self.render = render
        self.seed = seed

        self.actions = list(range(self.env.action_space.n))

        self.env.seed(seed)
        self.reset()

    def get_current_state(self):
        return self.curr_state

    def get_current_ram(self):
        return self.env.env.ale.getRAM()

    def execute_agent_action(self, action):
        image, _, _, _ = self.env.step(action)
        ram = self.get_current_ram()

        # if self.render:
        #     self.env.render()

        # reward = np.sign(reward)  # Reward clipping
        # is_dead = self.get_num_lives(ram) < self.num_lives
        # skull_direction = int(self.skull_pos > self.get_skull_pos(ram))

        self.curr_state = MRRAMState(ram, image)

        return 0 # TODO change later

    def sparse_gc_reward_function(self, state, goal, info={}, tol=2):
        assert isinstance(state, MRRAMState), type(state)
        assert isinstance(goal, MRRAMState), type(goal)

        def is_close(pos1, pos2):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        reached = is_close(state.get_position(), goal.get_position())
        reward = +1. if reached else 0.

        return reward, reached

    def reset(self):
        self.env.reset()
        self.remove_skull()

        for _ in range(4):
            image, _, _, _ = self.env.step(0)  # no-op to get agent onto ground

        ram = self.env.env.ale.getRAM()
        # self.skull_pos = self.get_skull_pos(ram)
        # skull_direction = int(self.skull_pos > self.get_skull_pos(ram))

        self.curr_state = MRRAMState(ram, image)

    def state_space_size(self):
        return self.curr_state.features().shape[0]

    def action_space_size(self):
        return len(self.actions)

    def sample_random_action(self):
        return random.choice(self.actions)

    @staticmethod
    def get_num_lives(ram):
        return int(MRRAMState.getByte(ram, 'ba'))

    @staticmethod
    def get_skull_pos(ram):
        skull_x = int(MRRAMState.getByte(ram, 'af')) + 33
        return skull_x / 100.

    def set_player_position(self, x, y):
        state_ref = self.env.env.ale.cloneState()
        state = self.env.env.ale.encodeState(state_ref)
        self.env.env.ale.deleteState(state_ref)
        
        state[331] = x
        state[335] = y
        
        new_state_ref = self.env.env.ale.decodeState(state)
        self.env.env.ale.restoreState(new_state_ref)
        self.env.env.ale.deleteState(new_state_ref)
        self.execute_agent_action(0) # NO-OP action to update the RAM state

    def remove_skull(self):
        state_ref = self.env.env.ale.cloneState()
        state = self.env.env.ale.encodeState(state_ref)
        self.env.env.ale.deleteState(state_ref)

        state[431] = 1
        state[351] = 390  # 40

        new_state_ref = self.env.env.ale.decodeState(state)
        self.env.env.ale.restoreState(new_state_ref)
        self.env.env.ale.deleteState(new_state_ref)
        self.execute_agent_action(0)  # NO-OP action to update the RAM state
    
    def saveImage(self, path):
        cv2.imwrite(f"{path}.png", self.curr_state.image[:,:,-1])