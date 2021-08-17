from gym import Wrapper

from hrl.option.utils import get_player_position


class MonteAgentSpace(Wrapper):
	"""
	crops out the surrounding pixels of the monte agent
	"""
	def __init__(self, env, width=20, height=24):
		self.env = env
		self.width = width
		self.height = height
	
	def get_pixels_around_player(self, image, width=20, height=24):
		"""
		given an image in monte, crop out just the player and its surroundings
		"""
		value_to_index = lambda x: int(-1.01144971 * x + 309.86119429)  # conversion from player position to pixel index
		player_position = get_player_position(self.env.unwrapped.ale.getRAM())
		start_x, end_x = (max(0, player_position[0] - width), 
							player_position[0] + width)
		start_y, end_y = (value_to_index(player_position[1]) - height,
							value_to_index(player_position[1]) + height)
		start_y += 0
		end_y += 8
		image_window = image[start_y:end_y, start_x:end_x, :]
		return image_window
	
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
