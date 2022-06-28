import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import animation
import os 
import glob

class VideoGenerator():

    def __init__(self,
            base_path):
        
        self.base_path = base_path
        self.img_path = os.path.join(self.base_path, 'tmp')

    def save_env_image(self, img, title):

        img = np.squeeze(img)

        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        i=0
        while os.path.exists(os.path.join(self.img_path, "{}.png".format(i))):
            i += 1

        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)

        fig.savefig(os.path.join(self.img_path, "{}.png".format(i)), bbox_inches="tight")

        plt.close("all")

    def create_video(self, episode):

        images = []

        for file in sorted(glob.glob(os.path.join(self.img_path, '*.png')), key=lambda f: int(''.join(filter(str.isdigit, f)))):
            images.append(file)

        fig = plt.figure(num=1, clear=True)

        ax = fig.add_subplot()

        patch = ax.imshow(imread(images[0]))
        ax.axis('off')

        def animate(idx):
            patch.set_data(imread(images[idx]))

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(images), interval=100)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # video_file = os.path.join(self.base_path, "ep_{}.gif".format(episode))
        video_file = os.path.join(self.base_path, "ep_{}.mp4".format(episode))

        # anim.save(video_file)
        anim.save(video_file, writer=animation.FFMpegWriter(fps=20))

        plt.close(fig)

        self.clear_images()

    def clear_images(self):

        if os.path.exists(self.img_path):
            for file in glob.glob(os.path.join(self.img_path, "*.png")):
                os.remove(file)

            os.rmdir(self.img_path)