import numpy as np
from gym import Wrapper
from gym.spaces.box import Box

from hrl.option.utils import get_player_position


class MonteAgentSpace(Wrapper):
	"""
	crops out the surrounding pixels of the monte agent
	"""
	def __init__(self, env, width=20, height=24):
		super().__init__(env)
		self.env = env
		self.width = width
		self.height = height
		self.y_offset = 8
		# by default, the observation shape is (56, 40, 3)
		self.img_shape = (2 * height + self.y_offset, 2 * width, 3)
		self.observation_space = Box(0, 255, self.img_shape)
	
	def get_pixels_around_player(self, image):
		"""
		given an image in monte, crop out just the player and its surroundings
		"""
		value_to_index = lambda x: int(-1.01144971 * x + 309.86119429)  # conversion from player position to pixel index
		player_position = get_player_position(self.env.unwrapped.ale.getRAM())
		start_x, end_x = (max(0, player_position[0] - self.width), 
							player_position[0] + self.width)
		start_y, end_y = (value_to_index(player_position[1]) - self.height,
							value_to_index(player_position[1]) + self.height)
		start_y += 0
		end_y += self.y_offset
		image_window = image[start_y:end_y, start_x:end_x, :]
		try:
			assert image_window.shape == self.img_shape
			return image_window
		except AssertionError:
			# the agent is going near the edges of the game
			# so start_x = max(0, player_p0s[0]) = 0
			# and the x_len is shorter than normal
			# we pad the image with black space to ensure the same shape
			patch_x = int(self.img_shape[1] - image_window.shape[1])
			padded_image_window = np.pad(image_window, pad_width=[(0, 0),(patch_x, 0),(0, 0)], mode='constant')
			assert padded_image_window.shape == self.img_shape, padded_image_window.shape
			return padded_image_window
	
	def reset(self):
		state = self.env.reset()
		cropped_state = self.get_pixels_around_player(state)
		return cropped_state
	
	def step(self, action):
		next_state, reward, done, info = self.env.step(action)
		cropped_next_state = self.get_pixels_around_player(next_state)
		return cropped_next_state, reward, done, info
	
	def render(self, mode="human"):
		img = self.env.unwrapped._get_image()
		img = self.get_pixels_around_player(img)
		if mode == "rgb_array":
			return img
		elif mode == "human":
			from gym.envs.classic_control import rendering

			if self.env.viewer is None:
				self.env.viewer = rendering.SimpleImageViewer()
			self.env.viewer.imshow(img)
			return self.env.viewer.isopen
