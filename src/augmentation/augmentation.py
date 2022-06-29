import os

import PIL.features
from PIL import Image
from pathlib import Path

"""
Usage:
All images in a folder named 'input' are transformed and saved in 'output'.
Each input image results in 4 output images (original, rotation by 180, original flipped, rotated flipped)
"""

# pillow version 9.1.1 or higher needed fore transpose
# conda install -c conda-forge pillow
# use conda-forge for community version which currently has a higher version
# print(PIL.features.pilinfo())

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for filename in os.listdir(INPUT_FOLDER):
    load_path = os.path.join(INPUT_FOLDER, filename)

    with Image.open(load_path) as im:
        im = PIL.ImageOps.exif_transpose(im)
        # im.load()

        new_filename = Path(filename).stem + "_" + "ORIGINAL" + ".jpg"
        im.save(os.path.join(OUTPUT_FOLDER, new_filename))

        im_transposed = im.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        new_filename = Path(filename).stem + "_" + "FLIP_LEFT_RIGHT" + ".jpg"
        im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        im_transposed = im.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        new_filename = Path(filename).stem + "_" + "FLIP_TOP_BOTTOM" + ".jpg"
        im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        #im_transposed = im.transpose(method=Image.Transpose.ROTATE_90)
        #new_filename = Path(filename).stem + "_" + "ROTATE_90" + ".jpg"
        #im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        im_transposed = im.transpose(method=Image.Transpose.ROTATE_180)
        new_filename = Path(filename).stem + "_" + "ROTATE_180" + ".jpg"
        im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        #im_transposed = im.transpose(method=Image.Transpose.ROTATE_270)
        #new_filename = Path(filename).stem + "_" + "ROTATE_270" + ".jpg"
        #im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        #im_transposed = im.transpose(method=Image.Transpose.TRANSPOSE)
        #new_filename = Path(filename).stem + "_" + "TRANSPOSE" + ".jpg"
        #im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        #im_transposed = im.transpose(method=Image.Transpose.TRANSVERSE)
        #new_filename = Path(filename).stem + "_" + "TRANSVERSE" + ".jpg"
        #im_transposed.save(os.path.join(OUTPUT_FOLDER, new_filename))

        #im_rotated = im.rotate(45)
        #new_filename = Path(filename).stem + "_" + str(45) + ".jpg"  # add degree of rotation to filename
        #im_rotated.save(os.path.join(OUTPUT_FOLDER, new_filename))

        print(f"Done: {filename}")
        im.close()

print("\nFinished")
