import os

import PIL.features
from PIL import Image
from pathlib import Path

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for filename in os.listdir(INPUT_FOLDER):
    load_path = os.path.join(INPUT_FOLDER, filename)

    with Image.open(load_path) as im:
        im = PIL.ImageOps.exif_transpose(im)
        # im.load()


        print(f"Done: {filename}")
        im.close()

print("\nFinished")
