from collections import deque
import glob
import os
from PIL import Image, ImageOps
from pfrl.wrappers.atari_wrappers import LazyFrames
import numpy as np

from tqdm import tqdm

def load_trajectory(data_dir, trajectory_num, stack_num=4):

    trajectory_dir = os.path.join(data_dir, 'trajectories', 'revenge', '{}.txt'.format(trajectory_num))
    screen_dir = os.path.join(data_dir, 'screens', 'revenge', '{}'.format(trajectory_num))

    if not os.path.exists(screen_dir):
        print("Trajectory not found at: {}".format(screen_dir))
        return

    print("Loading trajectory from: {}".format(screen_dir))

    frames = deque([], maxlen=stack_num)
    transition = {}
    transition["reward"] = 0
    trajectory = []
    count = 0
    previous_state = None

    with open(trajectory_dir) as f:
        for i, line in tqdm(enumerate(f)):
            if i > 1:
                data = line.rstrip('\n').replace(" ","").split(",")
                transition["reward"] += int(data[1])
                transition["is_state_terminal"] = (data[3] == "True")
                transition["action"] = int(data[4])
                frames.append(
                    get_image(os.path.join(screen_dir, '{}.png'.format(data[0])))
                )

                count += 1

                if count == stack_num or transition["is_state_terminal"]:
                    state = LazyFrames(list(frames), stack_axis=0)
                    if i != 2:
                        transition["next_state"] = state
                        transition["state"] = previous_state
                        trajectory.append(transition)
                                            
                    previous_state = state
                    transition["reward"] = 0
                    count = 0
                    
    return trajectory


def get_image(image_file, resize=(84,84)):
    with Image.open(image_file) as image:
        image = ImageOps.grayscale(image)
        image = image.resize(resize)
        image = np.array(image)

    image = np.expand_dims(image, axis=0)

    return image

if __name__ == "__main__":

    trajectory = load_trajectory(
        os.path.expanduser("~/Documents/research/code/montezuma/atari_v1"),
        1
    )

    print(len(trajectory))
    print(trajectory[0])