import gym
import ipdb
import numpy as np
from .wrappers import Wrapper

# TODO: This only works for the 1st room
START_POSITION = (77, 235)


class MontezumaInfoWrapper(Wrapper):
    def __init__(self, env):
        self.T = 0
        self.num_lives = None
        Wrapper.__init__(self, env)
    
    def reset(self, **kwargs):
        s0 = self.env.reset(**kwargs)
        self.num_lives = self.get_num_lives(self.get_current_ram())
        info = self.get_current_info(info={})
        return s0, info

    def step(self, action, clf):
        self.T += 1
        obs, reward, done, info = self.env.step(action, clf)
        info = self.get_current_info(info=info)
        self.num_lives = info["lives"]
        return obs, reward, done, info

    def get_current_info(self, info, update_lives=False):
        ram = self.get_current_ram()
    
        info["lives"] = self.get_num_lives(ram)
        info["player_x"] = self.get_player_x(ram)
        info["player_y"] = self.get_player_y(ram)
        info["has_key"] = self.get_has_key(ram)
        info["room_number"] = self.get_room_number(ram)
        info["jumping"] = self.get_is_jumping(ram)
        info["dead"] = self.get_is_player_dead(ram)
        info["falling"] = self.get_is_falling(ram)
        info["uncontrollable"] = self.get_is_in_non_controllable_state(ram)
        info["buggy_state"] = self.get_is_climbing_imaginary_ladder(ram)

        if update_lives:
            self.num_lives = info["lives"]

        return info

    def get_current_position(self):
        ram = self.get_current_ram()
        return self.get_player_x(ram), self.get_player_y(ram)

    def get_player_x(self, ram):
        return int(self.getByte(ram, 'aa'))

    def get_player_y(self, ram):
        return int(self.getByte(ram, 'ab'))

    def get_num_lives(self, ram):
        return int(self.getByte(ram, 'ba'))
    
    def get_is_falling(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_is_jumping(self, ram):
        return int(self.getByte(ram, 'd6') != 0xFF)

    def get_room_number(self, ram):
        return int(self.getByte(ram, '83'))

    def get_has_key(self, ram):
        return int(self.getByte(ram, 'c1')) != 0

    def get_current_ale(self):
        # return self.env.unwrapped.ale
        return self.env.environment.ale

    def get_current_ram(self):
        return self.get_current_ale().getRAM()

    @staticmethod
    def _getIndex(address):
        assert type(address) == str and len(address) == 2 
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row*16+col

    @staticmethod
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = MontezumaInfoWrapper._getIndex(address)
        return ram[idx]

    def get_player_status(self, ram):
        status = self.getByte(ram, '9e')
        status_codes = {
            0x00: 'standing',
            0x2A: 'running',
            0x3E: 'on-ladder',
            0x52: 'climbing-ladder',
            0x7B: 'on-rope',
            0x90: 'climbing-rope',
            0xA5: 'mid-air',
            0xBA: 'dead',  # dive 1
            0xC9: 'dead',  # dive 2
            0xC8: 'dead',  # dissolve 1
            0xDD: 'dead',  # dissolve 2
            0xFD: 'dead',  # smoke 1
            0xE7: 'dead',  # smoke 2
        }
        return status_codes[status]

    def get_is_player_dead(self, ram):
        player_status = self.get_player_status(ram)
        dead = player_status == "dead"
        time_to_spawn = self.getByte(ram, "b7")
        respawning = time_to_spawn > 0
        return dead or respawning

    def get_is_in_non_controllable_state(self, ram):
        player_status = self.get_player_status(ram)
        return self.get_is_jumping(ram) or \
            player_status in ("mid-air") or\
            self.get_is_falling(ram) or \
            self.get_is_player_dead(ram)

    def get_is_climbing_imaginary_ladder(self, ram):
        imaginary = self.get_player_x(ram) == 128
        ladder = self.get_player_status(ram) == "climbing-ladder"
        return imaginary and ladder
