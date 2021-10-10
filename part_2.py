try:
    import os
    import json
    import glob
    import argparse
    import random

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import scipy.misc
    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

K = 3


def pick_k_coordinates(lst, k):
    return None if not len(lst) else tuple(lst[i] for i in range(0, len(lst), len(lst) // k))


def pick_k_random_coordinates(lst, k):
    return None if not len(lst) else tuple(lst[i] for i in random.sample(range(0, len(lst)), k))


def find_pixels(img, real_image):
    lights = np.where(img == 19)
    lights = list(zip(lights[0], lights[1]))
    non_lights = np.where(img != 19)
    non_lights = list(zip(non_lights[0], non_lights[1]))
    coordinates_lights = pick_k_coordinates(lights, K)
    if coordinates_lights:
        cut_image(real_image, [coordinates_lights, pick_k_random_coordinates(non_lights, K)])
    # return [coordinates_lights, pick_k_random_coordinates(non_lights, K)] if coordinates_lights else []


def get_files_from_base(path, file_suffix):
    print(path, file_suffix)
    s = glob.glob(os.path.join(path, file_suffix))
    print(s)
    return s


def write_to_bin_file(value, filename):
    with open(f"./data_bin/{filename}.bin", 'wb') as file:
        file.write((value).to_bytes(24, byteorder='big', signed=False))


def write_image_to_bin_file(image, filename):
    image.astype('uint8').tofile(f"./data_bin/{filename}.bin")


def cut_image(img, pixels):
    bord_img = add_border(img)
    for coordinate in pixels[0]:
        x = coordinate[0]
        y = coordinate[1]
        write_to_bin_file(1, 'aachen_val')
        write_image_to_bin_file(bord_img[x - 40:x + 40, y - 40:y + 40, :], 'aachen_img')

    for coordinate in pixels[1]:
        x = coordinate[0]
        y = coordinate[1]
        write_to_bin_file(0, 'aachen_val')
        write_image_to_bin_file(bord_img[x - 40:x + 40, y - 40:y + 40, :], 'aachen_img')


def add_border(image):
    length, width, layers = image.shape
    length_array, width_array = np.zeros((length + 80) * 40 * 3).reshape((length + 80, 40, 3)), np.zeros(
        40 * width * 3).reshape((40, width, 3))
    image_full = np.vstack([image, width_array])
    image_full = np.vstack([width_array, image_full])
    image_full = np.hstack([length_array, image_full])
    image_full = np.hstack([image_full, length_array])
    return image_full


def main(argv=None):
    label_bases = [('./gtFine/train/aachen', '*_labelIds.png'), ('./gtFine/val', '*_labelIds.png')]
    img_bases = [('./leftImg8bit/train/aachen', '*_leftImg8bit.png'), ('./leftImg8bit/val', '*_leftImg8bit.png')]
    for i, path in enumerate(label_bases[0:1]):
        flist_1 = get_files_from_base(path[0], path[1])

    for i, path in enumerate(img_bases[0:1]):
        flist_2 = get_files_from_base(path[0], path[1])

    for index, image in enumerate(flist_1[:2]):
        print("f", flist_1)
        img_1 = np.array(Image.open(image)).astype('uint8')
        img_2 = np.array(Image.open(flist_2[index])).astype('uint8')
        find_pixels(img_1, img_2)
        plt.figure().clf()
        plt.imshow(img_1)
        plt.show(block=True)

def read_files(root_dir):

    files = glob.glob(root_dir + '/**/*.json', recursive=True)
    return files





if __name__ == '__main__':
    read_files('gtFine')
