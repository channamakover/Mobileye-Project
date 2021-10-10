
try:
    import os
    import json
    import glob
    import argparse
    import time

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import scipy.misc
    from PIL import Image

except ImportError:
    print("Need to fix the installation")
    raise

KERNEL_SIZE = 31
RADIUS = 50
IMAGE_TOP_CROP = 50
IMAGE_BOTTOM_CROP = 800
RED_RANGE_MIN = 100
RED_RANGE_MAX = 220
GREEN_RANGE_MIN = 160
GREEN_RANGE_MAX = 220
CONVOLVE_JUMPER = 3


def get_distinct_coordinates_dec(func):
    """decorator function for get_distinct_coordinates"""
    def wrapper(*args, **kwargs):
        func(wrapper.points, *args, **kwargs)
    wrapper.points = {}
    return wrapper


@get_distinct_coordinates_dec
def get_distinct_coordinates(points, x_array, y_array):
    """calculates and remove all points in the same circle radius
    :param points - points which don't have other point in their circle
    :param x_array x's values of the points to check
    :param y_array y's values of the points to check"""
    i = 0
    while i < (len(x_array)):
        flag = True
        for point_x, point_y in points.items():
            is_exist = is_in_radius((x_array[i], y_array[i]), (point_x, point_y), RADIUS)
            if is_exist:
                del x_array[i]
                del y_array[i]
                i -= 1
                flag = False
                break
        if flag:
            points[x_array[i]] = y_array[i]
        i += 1


def find_relevant_points(red_x, red_y, green_x, green_y):
    """return distinct points from the green and red points"""
    get_distinct_coordinates(green_x, green_y)
    get_distinct_coordinates(red_x, red_y)
    return red_x, red_y, green_x, green_y


def is_in_radius(point, point_center, radius):
    """calculates if point is in the circle of other point"""
    if abs(point[0] - point_center[0]) < radius and abs(point[1] - point_center[1]) < radius:
        return True
    return False


def convolve_layer(layer, kernel):
    """get layer of the image, convolve it
    :return convolution result, max local points"""
    layer_res = sg.convolve2d(layer[:, ::CONVOLVE_JUMPER], kernel, mode="same")
    filtered_red_res = maximum_filter(layer_res, size=100) == layer_res
    x, y = np.where(filtered_red_res)
    return layer_res, x, y


def get_points_in_range(res, x_array, y_array, range_min, range_max):
    """calculates the range of the points
    :return the point which suit the range"""
    x, y = [], []
    for i in range(len(x_array)):
        for j in range(len(y_array)):
            if range_min < res[y_array[i]][x_array[j]] < range_max:
                x.append(x_array[j] * CONVOLVE_JUMPER)
                y.append(y_array[i])
    return x, y


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    c_image = c_image.astype(np.float32)[IMAGE_TOP_CROP:IMAGE_BOTTOM_CROP, :, :]

    kernel = np.full((KERNEL_SIZE, KERNEL_SIZE), -1 / KERNEL_SIZE ** 2)
    kernel[KERNEL_SIZE // 2][KERNEL_SIZE // 2] = (KERNEL_SIZE ** 2 - 1) / KERNEL_SIZE ** 2

    # convolve red layer
    red_res, y1, x1 = convolve_layer(c_image[:, :, [0]].reshape(len(c_image), len(c_image[0])), kernel)
    green_res, y2, x2 = convolve_layer(c_image[:, :, [1]].reshape(len(c_image), len(c_image[1])), kernel)

    # convolve green layer
    x_red, y_red = get_points_in_range(red_res, x1, y1, RED_RANGE_MIN, RED_RANGE_MAX)
    x_green, y_green = get_points_in_range(green_res, x1, y1, GREEN_RANGE_MIN, GREEN_RANGE_MAX)

    # remove duplicate points in the same area
    red_x, red_y, green_x, green_y = find_relevant_points(x_red, y_red, x_green, y_green)
    return red_x, [y + IMAGE_TOP_CROP for y in red_y], green_x, [y + IMAGE_TOP_CROP for y in green_y]


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image.astype("uint8"))
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask







def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)

def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    s = time.time()
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = './data'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist[20:21]:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    f = time.time()
    print("total time = {}".format(f - s))
    plt.show(block=True)


if __name__ == '__main__':
    main()
