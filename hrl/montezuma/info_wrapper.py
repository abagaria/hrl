import gym


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        self.T = 0
        self.num_lives = None
        gym.Wrapper.__init__(self, env)
    
    def reset(self, **kwargs):
        s0 = self.env.reset(**kwargs)
        self.num_lives = self.get_num_lives(self.get_current_ram())
        info = self.get_current_info()
        return s0, info

    def step(self, action):
        self.T += 1
        obs, reward, done, info = self.env.step(action)
        info = self.get_current_info()
        self.num_lives = info["lives"]
        return obs, reward, done, info

    def get_current_info(self):
        ram = self.get_current_ram()
        
        info = {}
        info["lives"] = self.get_num_lives(ram)
        info["falling"] = self.get_is_falling(ram)
        info["player_x"] = self.get_player_x(ram)
        info["player_y"] = self.get_player_y(ram)
        info["dead"] = int(info["lives"] < self.num_lives)

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

    def get_current_ale(self):
        return self.env.unwrapped.ale

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
