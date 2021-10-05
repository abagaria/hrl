import gym
import numpy as np

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
    _is_cv2_available = True
except Exception:
    _is_cv2_available = False


class GreyScaleFrame(gym.ObservationWrapper):
    """
    converting RGB to grey-scale image and also divide by 255 to convert 
    observations from int8 to float32

    this wrapper should be used after MonteAgentSpaceForwarding, because in case
    the saved forwarding state is an image with 3 channels, the reset() method in
    this class takes care of that
    """
    def __init__(self, env):
        if not _is_cv2_available:
            raise RuntimeError(
                "Cannot import cv2 module. Please install OpenCV-Python to use"
                " MonteDownSampleFrames."
            )
        gym.ObservationWrapper.__init__(self, env)
        old_obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_obs_shape[0], old_obs_shape[1], 1),
            dtype=np.uint8,
        )
    
    def observation(self, observation):
        obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)  # this makes 3-dim -> 2-dim
        obs = obs.reshape((obs.shape[0], obs.shape[1], 1))  # make it 3-dim again
        return obs
    
    def reset(self):
        """
        used to convert 3-channel images into gray scale
        """
        obs = self.env.reset()
        assert len(obs.shape) == 3
        if obs.shape[-1] == 1:
            # already 1-channel, no need to do any work
            return obs
        elif obs.shape[-1] == 3:
            # convert RGB to grey scale
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # this makes 3-dim -> 2-dim
            obs = obs.reshape((obs.shape[0], obs.shape[1], 1))  # make it 3-dim again
            return obs
        else:
            raise ValueError('channel should be either 1 or 3')
