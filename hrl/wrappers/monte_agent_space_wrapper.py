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
