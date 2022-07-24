import os
import PIL.features
from PIL import Image

"""
All the images inside a sub folder in LOAD_FOLDER are scaled to 128x128 pixels and saved in 
the same sub folder in a folder named SAVE_FOLDER.
"""

# https://pillow.readthedocs.io/en/latest/reference/ImageOps.html#PIL.ImageOps.fit
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
# https://pillow.readthedocs.io/en/stable/reference/Image.html#the-image-class

LOAD_FOLDER = "images_raw"
SAVE_FOLDER = "images_resized"

# https://stackoverflow.com/questions/49882682/how-do-i-list-folder-in-directory
filenames = os.listdir(LOAD_FOLDER) # get all files' and folders' names in the directory
print("Loading following folders: " + str(filenames))

print("Working on following images:\n")
for index, subFolder in enumerate(filenames):
    subFOlderPath = os.path.join(LOAD_FOLDER, subFolder)
    for filename in os.listdir(subFOlderPath):
        load_path = os.path.join(subFOlderPath, filename)

        with Image.open(load_path) as im:
            im = PIL.ImageOps.exif_transpose(im)

            im_fitted = PIL.ImageOps.fit(image=im, size=(128, 128), centering=(0.5, 0.5))

            save_path = os.path.join(SAVE_FOLDER, subFolder, filename)
            if not os.path.exists(os.path.join(SAVE_FOLDER, subFolder)):
                os.makedirs(os.path.join(SAVE_FOLDER, subFolder))

            im_fitted.save(save_path)
            print(filename)

print("\nFinished")
