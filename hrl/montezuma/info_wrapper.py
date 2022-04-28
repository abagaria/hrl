import gym
import ipdb
import numpy as np
from .wrappers import Wrapper

# TODO: This only works for the 1st room
START_POSITION = (77, 235)


class MontezumaInfoWrapper(Wrapper):
    def __init__(self,
                env,
                use_persistent_death_flag=True,
                use_persistent_falling_flag=True):
        self.T = 0
        self.num_lives = None
        self.death_flag = False
        self.falling_flag = False
        self.falling_time = None
        self.use_persistent_death_flag = use_persistent_death_flag
        self.use_persistent_falling_flag = use_persistent_falling_flag
        Wrapper.__init__(self, env)
    
    def reset(self, **kwargs):
        self.falling_time = None
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
        
        # Did the player die in the current frame
        # (commented out portion is the animation flag from RAM, which is imperfect)
        current_death = int(info["lives"] < self.num_lives) # or (self.getByte(ram, 'b7') > 0)
        current_falling = self.get_is_falling(ram)

        # Raise the death flag
        if current_death:
            self.death_flag = True

        if current_falling:
            self.falling_flag = True
            self.falling_time = self.T

        # This takes the variable length of the death animation into account: 
        # Lower the death flag if the player has respawned at the start location
        if self.death_flag and self.get_current_position() == START_POSITION:
            self.death_flag = False

        # Lower the falling flag when the player has returned to the start position
        # Sometimes the falling signal goes up and the player doesn't die, in those cases
        # we don't want the falling flag to remain high for the entire episode
        if self.falling_flag and \
            (self.get_current_position() == START_POSITION or \
             self.falling_time - self.T >= 20 or \
             self.death_flag):
            self.falling_flag = False

        if self.use_persistent_death_flag:
            info["dead"] = self.death_flag
        else:
            info["dead"] = current_death

        if self.use_persistent_falling_flag:
            info["falling"] = self.falling_flag
        else:
            info["falling"] = current_falling

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
