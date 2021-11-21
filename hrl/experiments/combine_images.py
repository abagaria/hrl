import cv2
import numpy as np
import itertools
from pprint import pprint
import os

# spawns = [(77, 235), (130, 192), (123, 148), (20, 148), (21, 192), (114,148), (99, 148), (75, 148), (62, 148), (50, 148), (38, 148), (25, 148)]
import os
filelist=os.listdir('.')
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".png")):
        filelist.remove(fichier)
# print(filelist)

new_list = [os.path.splitext(x)[0] for x in filelist if len(x) > 15]
print(new_list)

# delete = 0

# for start, end in itertools.product(spawns, spawns):
#     print(start, end)
#     if start == end:
#         continue
#     img1 = cv2.imread(f'{start[0]},{start[1]}.png')
#     img2 = cv2.imread(f'{end[0]},{end[1]}.png')
#     # print(img1, img2)
#     im_v = cv2.vconcat([img1, img2])
#     # print(im_v.shape)
#     if delete:
#         os.remove(f'{start[0]},{start[1]}-{end[0]},{end[1]}.png')
#     else:
#         cv2.imwrite(f'{start[0]},{start[1]}-{end[0]},{end[1]}.png', im_v)