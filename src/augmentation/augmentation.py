import os
import torchvision
from torchvision import transforms
import PIL.features
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer

"""
All the images inside a sub folder in LOAD_FOLDER are scaled to 128x128 pixels and saved in 
the same sub folder in a folder named SAVE_FOLDER.
Old images are overwritten if a new image has the same name.
Old images are not deleted, therefore an old image may stay in the folder if it is not overwritten by a new one.
"""

# pillow version 9.1.1 or higher needed fore transpose
# conda install -c conda-forge pillow
# use conda-forge for community version which currently has a higher version
# print(PIL.features.pilinfo())

# https://pillow.readthedocs.io/en/latest/reference/ImageOps.html#PIL.ImageOps.fit
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
# https://pillow.readthedocs.io/en/stable/reference/Image.html#the-image-class

start = default_timer()

LOAD_FOLDER = "../assets/baseline/"
SAVE_FOLDER = "../assets/augmented/"
NUMBER_OF_AUGMENTED_IMAGES = 12

# https://stackoverflow.com/questions/49882682/how-do-i-list-folder-in-directory
filenames = os.listdir(LOAD_FOLDER) # get all files' and folders' names in the directory
print("Loading following folders: " + str(filenames))

print("Working on following images:\n")
for index, subFolder in enumerate(filenames):
    subFOlderPath = os.path.join(LOAD_FOLDER, subFolder)

    if not os.path.exists(os.path.join(SAVE_FOLDER, subFolder)):
        os.makedirs(os.path.join(SAVE_FOLDER, subFolder))

    for filename in os.listdir(subFOlderPath):
        image_number = 0
        load_path = os.path.join(subFOlderPath, filename)

        with Image.open(load_path) as im:
            im = PIL.ImageOps.exif_transpose(im)

            augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([128, 128]),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomRotation(degrees=20),
                #transforms.RandomCrop(128, 128),
                transforms.Grayscale(),
                #transforms.RandomPerspective(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(), # TODO do transform on RGB image then collorJitter
                transforms.Normalize((0.5), (0.5)), # TODO inplace?
                transforms.ToPILImage(),
            ])

            for i in range (0, NUMBER_OF_AUGMENTED_IMAGES):
                # TODO
                # Randomly black out areas
                # Add Gaussian noise

                # Random zoom
                # Flipping
                # shear

                '''
                fig, axs = plt.subplots(nrows=2, ncols=5)
                for i in range(0, 5):
                    for j in range(0, 2):
                        x = augmentation(im)
                        axs[j, i].plot(x)
                plt.show()
                '''

                im_augmented = augmentation(im)

                #im_fitted = PIL.ImageOps.fit(image=im, size=(128, 128), centering=(0.5, 0.5))

                new_filename = Path(filename).stem + "_" + str(image_number) + ".jpg"
                save_path = os.path.join(SAVE_FOLDER, subFolder, new_filename)

                im_augmented.save(save_path)
                print(new_filename)
                image_number = image_number + 1

duration = default_timer() - start
print("\nFinished")
print(duration)
