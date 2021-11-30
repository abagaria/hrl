import gym


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        self.T = 0
        self.num_lives = None
        gym.Wrapper.__init__(self, env)
    
    def reset(self, **kwargs):
        # s0 = self.env.reset(**kwargs)
        self.env.reset(**kwargs)
        self.remove_skull()
        s0, _, _, _ = self.env.step(0)
        self.num_lives = self.get_num_lives(self.get_current_ram())
        info = self.get_current_info(info={})
        return s0, info

    def step(self, action):
        self.T += 1
        obs, reward, done, info = self.env.step(action)
        info = self.get_current_info(info=info)
        self.num_lives = info["lives"]
        return obs, reward, done, info

    def get_current_info(self, info):
        ram = self.get_current_ram()
    
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

    def set_player_position(self, x, y):
        state_ref = self.env.env.ale.cloneState()
        state = self.env.env.ale.encodeState(state_ref)
        self.env.env.ale.deleteState(state_ref)
        
        state[331] = x
        state[335] = y
        
        new_state_ref = self.env.env.ale.decodeState(state)
        self.env.env.ale.restoreState(new_state_ref)
        self.env.env.ale.deleteState(new_state_ref)
        s0, _, _, _ = self.env.step(0) # NO-OP action to update the RAM state
        return s0

    def remove_skull(self):
        # print("Setting skull position")
        state_ref = self.get_current_ale().cloneState()
        state = self.get_current_ale().encodeState(state_ref)
        self.get_current_ale().deleteState(state_ref)

        state[431] = 1
        state[351] = 390  # 40

        new_state_ref = self.get_current_ale().decodeState(state)
        self.get_current_ale().restoreState(new_state_ref)
        self.get_current_ale().deleteState(new_state_ref)
        self.env.step(0)  # NO-OP action to update the RAM state

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
