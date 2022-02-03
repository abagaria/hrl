import gym

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info["player_x"] = self.get_player_x()
        info["player_y"] = self.get_player_y()
        info["room_number"] = self.get_room_number()
        info["has_key"] = self.get_has_key()

        return obs, reward, done, info

    def get_current_info(self):
        return dict(
            player_x=self.get_player_x(),
            player_y=self.get_player_y(),
            room_number=self.get_room_number(),
            has_key=self.get_has_key(),
            falling=self.get_is_falling()
        )

    def get_player_x(self):
        return int(self.getByte(self.get_current_ram(), 'aa'))

    def get_player_y(self):
        return int(self.getByte(self.get_current_ram(), 'ab'))

    def get_room_number(self):
        return int(self.getByte(self.get_current_ram(), '83'))

    def get_is_falling(self):
        return int(self.getByte(self.get_current_ram(), 'd8')) != 0

    def get_has_key(self):
        return int(self.getByte(self.get_current_ram(), 'c1')) != 0

    def get_current_ale(self):
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