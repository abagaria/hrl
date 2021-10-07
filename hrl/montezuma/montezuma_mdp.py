import copy
import random
from hrl.montezuma.montezuma_state import MontezumeState


class MontezumaMDP:
    def __init__(self, env, render):
        self.env = env
        self.render = render

        self.cur_state = None
        self.init_state = None
        self.actions = range(self.env.action_space.n)

        self.reset()

    def get_current_ale(self):
        return self.env.unwrapped.ale

    def get_current_ram(self):
        return self.get_current_ale().getRAM()

    def _transition_func(self, state, action):
        return self.next_state

    def _reward_func(self, state, action):
        reward, done, info = self._step(state, action)

        ram = self.get_current_ram()
        self.num_lives = self.cur_state.get_num_lives(ram)
        self.skull_pos = self.cur_state.get_skull_position(ram)

        return reward, done, info

    def _step(self, state, action):
        next_obs, reward, done, info = self.env.step(action)
        ram = self.get_current_ram()

        if self.render:
            self.env.render()

        is_dead = int(self.get_num_lives(ram) < self.num_lives)
        skull_direction = self.get_skull_direction(current_skull_pos=self.get_skull_pos(ram),
                                                   previous_skull_pos=self.skull_pos)

        self.next_state = MontezumeState(frame=next_obs,
                                         ram=ram,
                                         skull_direction=skull_direction,
                                         is_dead=is_dead,
                                         is_terminal=done)

        return reward, done, info

    def sparse_gc_reward_function(self, state, goal, info={}, tol=2):
        assert isinstance(state, MontezumeState), type(state)
        assert isinstance(goal, MontezumeState), type(goal)

        def is_close(pos1, pos2):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        reached = is_close(state.get_position(), goal.get_position())
        reward = +1. if reached else 0.

        return reward, reached

    def reset(self):
        obs = self.env.reset()

        # Set init state
        ram = self.get_current_ram()
        self.skull_pos = self.get_skull_pos(ram)
        skull_direction = self.get_skull_direction(self.skull_pos, self.skull_pos)
        self.init_state = MontezumeState(frame=obs,
                                         ram=ram,
                                         skull_direction=skull_direction,
                                         is_dead=0,
                                         is_terminal=0)
        self.num_lives = self.init_state.get_num_lives(ram)

        # Set current state
        self.cur_state = copy.deepcopy(self.init_state)

    def state_space_size(self):
        return self.init_state.features()

    def action_space_size(self):
        return len(self.actions)

    def sample_random_action(self):
        return random.choice(self.actions)

    @staticmethod
    def get_x_y_low_lims():
        return 0, 100

    @staticmethod
    def get_x_y_high_lims():
        return 140, 300

    @staticmethod
    def get_num_lives(ram):
        return int(MontezumeState.getByte(ram, 'ba'))

    @staticmethod
    def get_skull_pos(ram):
        skull_x = int(MontezumeState.getByte(ram, 'af')) + 33
        return skull_x

    @staticmethod
    def get_skull_direction(current_skull_pos, previous_skull_pos):
        return int(previous_skull_pos > current_skull_pos)

    def execute_agent_action(self, action):
        reward, done, info = self._reward_func(self.cur_state, action)
        next_state = self._transition_func(self.cur_state, action)
        self.cur_state = next_state

        return next_state, reward, done, info

    def remove_skull(self):
        print("Setting skull position")
        state_ref = self.get_current_ale().cloneState()
        state = self.get_current_ale().encodeState(state_ref)
        self.get_current_ale().deleteState(state_ref)

        state[431] = 1
        state[351] = 390  # 40

        new_state_ref = self.get_current_ale().decodeState(state)
        self.get_current_ale().restoreState(new_state_ref)
        self.get_current_ale().deleteState(new_state_ref)
        self.execute_agent_action(0)  # NO-OP action to update the RAM state
