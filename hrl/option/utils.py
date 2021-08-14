import gym
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)


def warp_frames(state):
	"""
	warp frames from (210, 160, 3) to (1, 84, 84) as in the nature paper
	this mimics the WarpFrame wrapper:
	https://github.com/pfnet/pfrl/blob/7b0c7e938ba2c0c56a941c766c68635d0dad43c8/pfrl/wrappers/atari_wrappers.py#L156
	"""
	size = (1, 84, 84)
	warped = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
	warped = cv2.resize(warped, (84, 84), interpolation=cv2.INTER_AREA)
	observation_space = gym.spaces.Box(
		low=0, high=255, shape=size, dtype=np.uint8
	)
	return warped.reshape(observation_space.low.shape)

def get_player_position(ram):
	"""
	given the ram state, get the position of the player
	"""
	def _getIndex(address):
		assert type(address) == str and len(address) == 2
		row, col = tuple(address)
		row = int(row, 16) - 8
		col = int(col, 16)
		return row * 16 + col
	def getByte(ram, address):
		# Return the byte at the specified emulator RAM location
		idx = _getIndex(address)
		return ram[idx]
	# return the player position at a particular state
	x = int(getByte(ram, 'aa'))
	y = int(getByte(ram, 'ab'))
	return x, y
