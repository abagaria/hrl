from pfrl.wrappers.atari_wrappers import ScaledFloatFrame

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
    _is_cv2_available = True
except Exception:
    _is_cv2_available = False


class MonteDownSampleFrames(ScaledFloatFrame):
    """
    down sample observations from monte by converting RGB to grey-scale image
    and also divide by 255 to convert observations from int8 to float32

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
        super().__init__(env)
    
    def observation(self, observation):
        obs = super().observation(observation)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # this makes 3-dim -> 2-dim
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
