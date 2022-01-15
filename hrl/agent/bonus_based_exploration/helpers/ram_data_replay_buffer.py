import os
from absl import logging
import numpy as np

class MontezumaRevengeReplayBuffer():
    def __init__(
        self,
        replay_capacity=1000000
    ):

        self.replay_capacity = replay_capacity
        self._create_storage()

    def _create_storage(self):
        self.memory = {}

        self.memory['player_x'] = np.empty([self.replay_capacity], dtype=np.int)
        self.memory['player_y'] = np.empty([self.replay_capacity], dtype=np.int)
        self.memory['room_number'] = np.empty([self.replay_capacity], dtype=np.int)

    def add(self, player_x, player_y, room_number, cursor):

        self.memory['player_x'][cursor] = player_x
        self.memory['player_y'][cursor] = player_y
        self.memory['room_number'][cursor] = room_number      

    def _generate_filename(self, checkpoint_dir, name):
        info_dir = os.path.join(checkpoint_dir, 'ram_info')
        if not os.path.isdir(info_dir):
            os.makedirs(info_dir)
        return os.path.join(info_dir, '{}_ckpt.npy'.format(name))

    def get_index(self, memory_type, index):
        return self.memory[memory_type][index]

    def get_indices(self, memory_type, indices):
        return self.memory[memory_type][indices]
    
    def save(self, checkpoint_dir):
        if not os.path.isdir(checkpoint_dir):
            return

        filename = self._generate_filename(checkpoint_dir, 'player_x')
        np.save(filename, self.memory['player_x'], allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'player_y')
        np.save(filename, self.memory['player_y'], allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'room_number')
        np.save(filename, self.memory['room_number'], allow_pickle=False)

    def load(self, checkpoint_dir):

        filename = self._generate_filename(checkpoint_dir, 'player_x')
        if os.path.exists(filename):
            logging.info('Loading player_x data')
            self.memory['player_x'] = np.load(filename, allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'player_y')
        if os.path.exists(filename):
            logging.info('Loading player_y data')
            self.memory['player_y'] = np.load(filename, allow_pickle=False)

        filename = self._generate_filename(checkpoint_dir, 'room_number')
        if os.path.exists(filename):
            logging.info('Loading room_number data')
            self.memory['room_number'] = np.load(filename, allow_pickle=False)

