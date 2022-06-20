import os
from PIL import Image
from pathlib import Path

"""
Usage:
All images (that PIL can read) in the folder "images_raw" are assumed to be in a 1:1 ratio.
These images will be cropped to a 3:2 ratio and are then scaled to 300x200 pixels.
The resulting images are saved in the folder "images_resized".
"""

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

    for i in range (0,4):
        im_rotated = im.rotate(i*90)
        im_cropped = im_rotated.crop((left, top, right, bottom))
        im_resized = im_cropped.resize((300, 200))

        new_filename = Path(filename).stem + "_" + str(i * 90) + ".jpg"     # add degree of rotation to filename
        save_path = os.path.join(SAVE_FOLDER, new_filename)
        im_resized.save(save_path)
        print(new_filename)

print("\nFinished")
