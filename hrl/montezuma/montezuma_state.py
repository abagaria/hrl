import numpy as np


class MontezumeState:
    def __init__(self, frame, ram, skull_direction, is_dead, is_terminal):
        """
        Constructor for MR RAM State.
        Args:
            frame: raw image observation from the env
            ram: ram state from the game engine
            skull_direction: 1 if right, else left
            is_dead: whether the player died in the current frame
            is_terminal: whether the game was over in the current frame
        """

        self.frame = frame
        self._is_terminal = is_terminal

        x = self.get_player_x(ram)
        y = self.get_player_y(ram)
        direction = self.get_direction(ram)
        lives = self.get_num_lives(ram)
        jumping = self.get_is_jumping(ram)
        falling = self.get_is_falling(ram)
        has_key = self.get_has_key(ram)
        skull_pos = self.get_skull_position(ram)
        skull_present = self.get_is_skull_present(ram)
        skull_dir = skull_direction
        left_door_locked = self.get_is_left_door_locked(ram)
        right_door_locked = self.get_is_right_door_locked(ram)
        self.is_dead = int(is_dead)

        self.ram_features = [x, y, direction, lives, jumping, falling, has_key, skull_pos,
                             skull_dir, skull_present, left_door_locked, right_door_locked, is_dead]

        self.position = self.get_position(ram)

    def __str__(self):
        return "x: {}, y: {}, dir: {}, lives: {}, jumping: {}, falling: {}, hasKey: {}, \
            skullPos: {}, skullDir: {}, skullPresent: {}, leftDoorLocked: {}, rightDoorLocked: {} \
            isDead: {}".format(*tuple(self.ram_features))

    def features(self):
        return self.frame

    def is_terminal(self):
        return self._is_terminal

    def get_ram_features(self):
        return self.ram_features

    def get_player_x(self, ram):
        return int(self.getByte(ram, 'aa'))

    def get_player_y(self, ram):
        return int(self.getByte(ram, 'ab'))

    def get_position(self, ram):
        x = self.get_player_x(ram)
        y = self.get_player_y(ram)
        return np.array((x, y))

    def get_direction(self, ram):
        # look is 1 if player is looking left else 0
        look = int(format(self.getByte(ram, 'b4'), '08b')[-4])
        return look

    def get_num_lives(self, ram):
        return int(self.getByte(ram, 'ba'))

    def get_has_key(self, ram):
        return int(int(self.getByte(ram, 'c1')) != 0)

    def get_is_jumping(self, ram):
        return 1 if self.getByte(ram, 'd6') != 0xFF else 0

    def get_is_falling(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_skull_position(self, ram):
        skull_x = int(self.getByte(ram, 'af')) + 33
        return skull_x

    def get_is_skull_present(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        return int(objects[2])

    def get_is_right_door_locked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        return int(objects[1])

    def get_is_left_door_locked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        return int(objects[0])

    @staticmethod
    def _getIndex(address):
        assert type(address) == str and len(address) == 2
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row * 16 + col

    @staticmethod
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = MontezumeState._getIndex(address)
        return ram[idx]

    def __repr__(self):
        return str(self)

    def __hash__(self):
        raise NotImplemented()

    def __eq__(self, other):
        raise NotImplemented(other)
