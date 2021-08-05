from enum import IntEnum

from gym import Wrapper


class actions(IntEnum):
    INVALID         = -1
    NOOP            = 0
    FIRE            = 1
    UP              = 2
    RIGHT           = 3
    LEFT            = 4
    DOWN            = 5
    UP_RIGHT        = 6
    UP_LEFT         = 7
    DOWN_RIGHT      = 8
    DOWN_LEFT       = 9
    UP_FIRE         = 10
    RIGHT_FIRE      = 11
    LEFT_FIRE       = 12
    DOWN_FIRE       = 13
    UP_RIGHT_FIRE   = 14
    UP_LEFT_FIRE    = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE  = 17


class MonteAgentSpace(Wrapper):
	"""
	crops out the surrounding pixels of the monte agent
	"""
	def __init__(self, env, width=20, height=24):
		self.env = env
		self.width = width
		self.height = height
	
	def get_state(self):
		"""
		get the current state
		"""
		ram = self.env.unwrapped.ale.getRAM()
		raise NotImplemented
	
	def get_pixels_around_player(self, image, width=20, height=24, trim_direction=actions.INVALID):
		"""
		given an image in monte, crop out just the player and its surroundings
		"""
		if trim_direction != actions.INVALID:
			width -= 6
		value_to_index = lambda x: int(-1.01144971 * x + 309.86119429)  # conversion from player position to pixel index
		# TODO:
		player_position = self.get_state()['player_x'], self.get_state()['player_y']  # - y_offset
		start_x, end_x = (max(0, player_position[0] - width), 
							player_position[0] + width)
		start_y, end_y = (value_to_index(player_position[1]) - height,
							value_to_index(player_position[1]) + height)
		start_y += 0
		end_y += 8
		if trim_direction == actions.RIGHT:
			start_x += 13
			end_x += 13
		elif trim_direction == actions.LEFT:
			start_x -= 7
			end_x -= 7
		image_window = image[start_y:end_y, start_x:end_x, :]
		return image_window
