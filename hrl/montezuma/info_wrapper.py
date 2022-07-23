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
        self.imaginary_ladder_locations = set()
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
        info["room_number"] = self.get_room_number(ram)
        info["jumping"] = self.get_is_jumping(ram)
        info["dead"] = self.get_is_player_dead(ram)
        info["falling"] = self.get_is_falling(ram)
        info["uncontrollable"] = self.get_is_in_non_controllable_state(ram)
        info["buggy_state"] = self.get_is_climbing_imaginary_ladder(ram)
        info["left_door_open"] = self.get_is_left_door_unlocked(ram)
        info["right_door_open"] = self.get_is_right_door_unlocked(ram)
        info["inventory"] = self.get_player_inventory(ram)

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

    def get_player_inventory(self, ram):
        # 'torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer'
        return format(self.getByte(ram, 'c1'), '08b')
    
    def get_is_falling(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_is_jumping(self, ram):
        return int(self.getByte(ram, 'd6') != 0xFF)

    def get_room_number(self, ram):
        return int(self.getByte(ram, '83'))

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
        # imaginary = self.get_player_x(ram) == 128
        screen = self.get_room_number(ram)
        x_pos = self.get_player_x(ram)
        y_pos = self.get_player_y(ram)
        position = x_pos, y_pos
        imaginary = not self.at_any_climb_region(position, screen)
        ladder = self.get_player_status(ram) == "climbing-ladder"
        climbing = imaginary and ladder
        known_imaginary = (x_pos, y_pos, screen) in self.imaginary_ladder_locations
        if climbing:
            print(f"Found climbing imaginary ladder {position, screen}")
        if climbing and not known_imaginary:
            print(f"New imaginary location, adding {x_pos, y_pos, screen} to blacklist")
            self.imaginary_ladder_locations.add((x_pos, y_pos, screen))
        return climbing or known_imaginary

    def get_is_left_door_unlocked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        left_door = objects[0]
        locked = int(left_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
        return not locked

    def get_is_right_door_unlocked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        right_door = objects[1]
        locked = int(right_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
        return not locked

    def at_any_climb_region(self, pos, screen):
        climb_margin = 4
        for x_center, ylim in self.get_climb_regions(screen):
            if (ylim[0] <= pos[1] <= ylim[1]
                    and abs(pos[0] - x_center) <= climb_margin):
                return True
        return False

    def get_climb_regions(self, screen):
        LEFT_LADDER = 0x14
        CENTER_LADDER = 0x4d
        RIGHT_LADDER = 0x85
        SCREEN_TOP = 0xfe
        UPPER_LEVEL = 0xeb
        MIDDLE_LEVEL_1 = 0xc0
        LOWER_LEVEL_1 = 0x94
        LOWER_LEVEL_5 = 0x9d
        LOWER_LEVEL_14 = 0xa0
        SCREEN_BOTTOM = 0x86

        regions = []
        climb_margin_y = 6
        
        # start-screen ladders
        if screen == 1:
            regions.append((CENTER_LADDER, (MIDDLE_LEVEL_1, UPPER_LEVEL)))
            regions.append((LEFT_LADDER, (LOWER_LEVEL_1, MIDDLE_LEVEL_1)))
            regions.append((RIGHT_LADDER, (LOWER_LEVEL_1, MIDDLE_LEVEL_1)))
        if screen == 5:
            regions.append((CENTER_LADDER, (SCREEN_BOTTOM, LOWER_LEVEL_5)))
            regions.append((CENTER_LADDER, (SCREEN_BOTTOM - 1, SCREEN_BOTTOM)))
        if screen == 14:
            regions.append((CENTER_LADDER, (SCREEN_BOTTOM, LOWER_LEVEL_14)))
            regions.append((CENTER_LADDER, (SCREEN_BOTTOM - 1, SCREEN_BOTTOM)))
        # tall bottom ladders
        if screen in [0, 2, 3, 4, 7, 11, 13]:
            regions.append((CENTER_LADDER, (SCREEN_BOTTOM, UPPER_LEVEL)))
            regions.append((CENTER_LADDER, (SCREEN_BOTTOM - 1, SCREEN_BOTTOM)))
        # short top ladders
        if screen in [4, 6, 9, 11, 13, 19, 21]:
            regions.append((CENTER_LADDER, (UPPER_LEVEL, SCREEN_TOP)))
            regions.append((CENTER_LADDER, (SCREEN_TOP, SCREEN_TOP + 1)))
        elif screen in [10, 22]:
            # add vertical landmark just above the bridge
            regions.append(
                (CENTER_LADDER, (UPPER_LEVEL, UPPER_LEVEL + climb_margin_y)))
            regions.append(
                (CENTER_LADDER, (UPPER_LEVEL + climb_margin_y, SCREEN_TOP)))
            regions.append((CENTER_LADDER, (SCREEN_TOP, SCREEN_TOP + 1)))
        return regions
