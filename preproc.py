from os import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output


def mask_data(what, index=100):
    """
    Radar maps are differently masked. Masks all maps with mask of chosen map.
    :param what: np array to be masked (3D, channels first)
    :param index: int, first dim. index. Mask of this map will be applied on each array.
    :return: np array of same size as input but uniformly masked
    """
    for i in range(len(what)):  # 237
        what[i][what[index] == -1] = -1
    return what


def generate_datasets(array, n=10, size=64, length=3, split=None):
    """
    Splits input array into training, cross validation and test sets.
    :param array: 3D np array, dataset to be split. This is a channel*900*900 large array of rain radar maps.
    Data is created by cutting size*size frames randomly from length consecutive map.
    :param n: int, total number of data instances to make
    :param size: int, height and width in pixels of each frame
    :param length: int, length in "time". Defines how many consecutive frames will belong to one instance.
    :param split: list or np array of either floats between 0 and 1 or positive integers. Set to None by deafult
    which means no splitting, just return all instances in one set.
    :return: 3D np array, either one dataset of smaller image frames or three datasets for training,
    cross validating and testing
    """
    ch = np.shape(array)[0]
    h = np.shape(array)[1]
    w = np.shape(array)[2]
    images = np.empty((n, length, size, size))  # n series, each consisting of length frames of size size**2
    for i in range(n):
        # draw 3 random numbers for map number and idx of top left pixel of window
        valid = 0
        while not valid:
            anchor = (np.random.randint(0, ch - length), np.random.randint(0, h - size), np.random.randint(0, w - size))
            # update_output(f"Cutting images...\n[{i+1}/{n}]")
            image = [ar[anchor[1]:anchor[1] + size,
                     anchor[2]:anchor[2] + size] for ar in np.asarray(array)[anchor[0]:anchor[0] + length]]
            valid = valid_image(image)
        images[i] = image

    txt = f"Shape of data: {np.shape(images)}"
    if split is not None:  # split
        if all((r <= 1) & (r >= 0) for r in split):
            assert (sum(split) == 1), "Split values must sum up to 1."
            train = images[:int(n * split[0])]
            xval = images[int(n * split[0]):int(n * (split[0] + split[1]))]
            test = images[int(n * (split[0] + split[1])):]
        elif all(isinstance(r, int) for r in split):
            train = images[:int(n * split[0] / sum(split))]
            xval = images[int(n * split[0] / sum(split)):int(n * (split[0] + split[1]) / sum(split))]
            test = images[int(n * (split[0] + split[1]) / sum(split)):]
        else:
            sys.exit("All split values must be either fractions for percentages or integers.")

        update_output(
            txt + f"\n\nTraining set: {np.shape(train)}\nValidation set: {np.shape(xval)}\nTest set: {np.shape(test)}")
        return train, xval, test
    else:  # no split
        update_output(txt)
        return images


def valid_image(image):
    """
    Filters out some useless data. in the junk variable several conditions are defined to check on the images.
    Currently it checks the number of different entry values and if 0s or 1s make up 0.75 part of the whole data.
    This throws out the cuts made inside or almost inside the mask region and rainless areas.
    Still can be improved.
    :param image: 3D np array, dimensions are the number of consecutive frames, height and width
    :return: bool, whether the data instance is valid in terms of usability
    """
    junk = [len(set(np.array(ar).flatten())) <= 4 for ar in image]
    junk += [len(np.array(image).flatten()[np.array(image).flatten() <= 0]) > 0.75 * len(np.array(image).flatten())]
    return 0 if any(junk) else 1


def update_output(string):
    """
    Utility method for logging. Automatically replaces last output.
    :param string: string to print out.
    """
    clear_output(wait=True)
    print(string)


def visualise_data(images, cmap='viridis', facecolor='w'):
    """
    Plots random elements e.g. from training dataset.
    :param images: 4D np array, image frames to plot. Dimensions: (#data in set, #frames, h, w)
    """

    num_img = np.shape(images)[0]
    n = np.random.randint(0, num_img)

    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor=facecolor)
    l = np.shape(images)[1]
    for i in range(l):
        plt.subplot(1, l, i + 1)
        plt.imshow(np.ma.masked_where(images[n][i] < 0, images[n][i]), cmap=cmap)
        plt.title(f"Instance #{n+1} from {num_img}\nFrame: {i}")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
