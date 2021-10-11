import numpy as np
from gym import Wrapper
from gym.spaces.box import Box

from hrl.option.utils import get_player_position
from hrl.wrappers.monte_agent_space_wrapper import MonteAgentSpace

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
    _is_cv2_available = True
except Exception:
    _is_cv2_available = False


class MonteDeepMindAgentSpace(MonteAgentSpace):
    """
    crops out the surrounding pixels of the monte agent
    This is specifically designed for cropping out pixels of a monte frame that
    has been through the wrap_deepmind wrappers, ans id of size (4, 84, 84)

    To crop out pixels of a normal monte frame of size (210, 160, 3), use MonteAgentSpace

    strategy for cropping:
        the monte ram state tells us the pixel location of the agent in a normal
        frame (210, 160, 3).
        We can then put a mask of 1's on the surrounding pixels of the agent on
        the (210, 160, 3) frame, and 0's everywhere else. 
        We warp this mask into shape (84, 84, 1), and the part the the mask that
        are 1's is the pixel area we eventually want
    """
    def __init__(self, env, width=20, height=24):
        """
        the width and height are pixel units relative to the original (210, 160, 3) frame
        """
        if not _is_cv2_available:
                    raise RuntimeError(
                        "Cannot import cv2 module. Please install OpenCV-Python to use"
                        " MonteDeepMindAgentSpace."
                    )
        Wrapper.__init__(self, env)
        self.env = env
        self.width = width
        self.height = height
        self.y_offset = 8
        self.img_shape = (4, 24, 24)
        self.observation_space = Box(0, 255, self.img_shape)
    
    def get_pixels_around_player(self, image):
        """
        given an image in monte, crop out just the player and its surroundings
        this image is a LazyFrame
        """
        value_to_index = lambda y: int(-1.01144971 * y + 309.86119429)  # move frame upwards
        player_position = get_player_position(self.env.unwrapped.ale.getRAM())  # pixel location of the player
        start_x, end_x = (max(0, player_position[0] - self.width), 
                            player_position[0] + self.width)
        start_y, end_y = (value_to_index(player_position[1]) - self.height,
                            value_to_index(player_position[1]) + self.height)
        start_y += 0
        end_y += self.y_offset

        # make the mask 
        mask = np.zeros(self.env.unwrapped.observation_space.low.shape)
        mask[start_y: end_y, start_x: end_x, :] = 1
        # sometimes the agent is going near the edges of the game, and so
        # start_x = max(0, player_p0s[0]) = 0, and x_len is shorter than normal
        # but here we don't need to worry about when the shape of the 1's is not self.img_shape
        # because we are automatically padding zeros

        # warp the mask
        def warp_frame(frame):
            """
            warp a frame from (210, 160, 3) to (84, 84, 1)
            """
            frame = np.mean(frame, axis=-1)
            frame = cv2.resize(
                frame, (84, 84), interpolation=cv2.INTER_AREA
            )   
            return frame.reshape((84, 84, 1))
        warped_mask = warp_frame(mask)
        warped_mask = warped_mask.squeeze()  # (84, 84)

        # find where in the warped_mask is a matrix of non 0's
        # don't find the matrix of 1's because the value 1 have been warped to another value
        x_coors, y_coors = np.where(warped_mask != 0)

        # use that matrix of 1's to crop out the final image
        cropped_image = []
        for frame in np.array(image):
            # image is an array of stacked frames
            cropped_frame = frame[x_coors[0]: x_coors[-1]+1, y_coors[0]: y_coors[-1]+1]
            padded_frame = np.zeros(self.img_shape[1:])
            padded_frame[:cropped_frame.shape[0], :cropped_frame.shape[1]] = cropped_frame
            cropped_image.append(padded_frame)
        cropped_image = np.array(cropped_image)
        return cropped_image
