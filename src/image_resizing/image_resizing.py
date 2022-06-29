import os
from PIL import Image
from pathlib import Path

"""
Usage:
All images (that PIL can read) in the folder "images_raw" are assumed to be in a 1:1 ratio.
These images will be cropped to a 3:2 ratio and are then scaled to 300x200 pixels.
The resulting images are saved in the folder "images_resized".
"""

# TODO
# check this new method, it should do exactly what we want
# https://pillow.readthedocs.io/en/latest/reference/ImageOps.html#PIL.ImageOps.fit

# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
# https://pillow.readthedocs.io/en/stable/reference/Image.html#the-image-class

LOAD_FOLDER = "images_raw"
SAVE_FOLDER = "images_resized"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

for filename in os.listdir(LOAD_FOLDER):
    load_path = os.path.join(LOAD_FOLDER, filename)
    im = Image.open(load_path)

    width, height = im.size
    offset = 2/3 * height / 2 / 2
    left = 0
    top = offset
    right = width
    bottom = height-offset

    im_cropped = im.crop((left, top, right, bottom))
    im_resized = im_cropped.resize((300, 200))
    save_path = os.path.join(SAVE_FOLDER, filename)
    im_resized.save(save_path)
    print(filename)

print("\nFinished")
