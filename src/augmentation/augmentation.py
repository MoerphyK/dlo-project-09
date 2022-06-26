TODO hier stehen nur codeschnipsel

for i in range(0, 1):
    im_rotated = im.rotate(i * 90)

    new_filename = Path(filename).stem + "_" + str(i * 90) + ".jpg"  # add degree of rotation to filename

    # im_rotated.save(os.path.join(SAVE_FOLDER, filename))