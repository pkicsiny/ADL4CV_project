from os import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
from matplotlib import colors

"""
DATA IMPORT
"""


def get_data(data_dir, total_length=None, mask=True):
    files = listdir(data_dir)
    if total_length is None:
        total_length = len(files)
    # array of inputs
    inputs = np.empty((total_length, 900, 900))
    # import data to the inputs array
    for i, file in enumerate(files):
        if i == total_length:
            break
        clear_output(wait=True)
        print(f"[{i+1}/{total_length}]")
        ascii_grid = np.loadtxt(f"{data_dir}/{files[i]}", skiprows=6)
        inputs[i] = ascii_grid
    if mask & (total_length > 100):
        inputs = mask_data(inputs, 100)
    return inputs

"""
PREPROCESSING
"""


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


def generate_datasets(array, n=10, size=64, length=3, split=None, normalize=False, task="prediction", down_factor=4):
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

    if task == "upsampling":
        length = 1
    elif task != "prediction":
        sys.exit("Task must be 'prediction' or 'upsampling'.")

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

    # normalization
    if normalize:
        images = images / images.max()

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

        txt = txt + f"\n\nTraining set: {np.shape(train)}\nValidation set: {np.shape(xval)}\nTest set: {np.shape(test)}"
        if task == "upsampling":  # downsample
            low_res_train, low_res_xval, low_res_test = [data[:, :, ::down_factor, ::down_factor] for data in
                                                         [train, xval, test]]
            update_output(txt + f"\n\nShape of downsampled data:\n\n" +
            f"Training set: {np.shape(low_res_train)}\nValidation set: {np.shape(low_res_xval)}\nTest set: {np.shape(low_res_test)}")
            return {"low_res_train": low_res_train,
                    "low_res_xval": low_res_xval,
                    "low_res_test": low_res_test,
                    "train": train,
                    "xval": xval,
                    "test": test}
        else:
            update_output(txt)
            return {"train": train, "xval": xval, "test": test}
    else:  # no split
        if task == "upsampling":  # downsample
            low_res_data = images[:, :, ::down_factor, ::down_factor]
            update_output(txt + f"\nShape of downsampled data: {np.shape(low_res_data)}")
            return {"low_res_data": low_res_data, "images": images}  # data and ground truth
        else:
            update_output(txt)
            return {"images": images}


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

"""
UTILS
"""


def update_output(string):
    """
    Utility method for logging. Automatically replaces last output.
    :param string: string to print out.
    """
    clear_output(wait=True)
    print(string)

"""
LOSS FUNCTIONS
"""


def gradient_diff(yTrue, yPred):
    alpha = 1
    return keras.backend.sum(keras.backend.pow(keras.backend.abs(keras.backend.abs(yTrue[:,:,1:,:,:] - yTrue[:,:,:-1,:,:]) -
           keras.backend.abs(yPred[:,:,1:,:,:] - yPred[:,:,:-1,:,:])),alpha)) + keras.backend.sum(
           keras.backend.pow(keras.backend.abs(keras.backend.abs(yTrue[:,:,:,1:,:] - yTrue[:,:,:,:-1,:]) -
           keras.backend.abs(yPred[:,:,:,1:,:] - yPred[:,:,:,:-1,:])),alpha))


def relative_error_tensor(truth, predictions):
    """
    :param truth: ground truth
    :param predictions: predictions of network
    :return: relative error(scalar)
    """
    results = tf.divide(keras.backend.sum(keras.backend.abs(tf.subtract(predictions, truth))),
                        keras.backend.sum(keras.backend.abs(truth)))
    print(results)
    print(truth)
    return results

"""
METRICS
"""


def relative_error(truth, predictions):
    """
    :param truth: ground truth
    :param predictions: predictions of network
    :return: relative error(array)
    """
    images = np.zeros_like(truth)
    sums = np.zeros(predictions.shape[0])  # sample size
    means = np.zeros(predictions.shape[0])
    for i in range(0, predictions.shape[0]):  # loop over samples
        num = np.abs(predictions[i, 0, :, :] - truth[i, 0, :, :])
        den = np.abs(truth[i, :, :, :])
        images[i, :, :, :] = np.divide(num, den)
        sums[i] = np.sum(num) / np.sum(den)
        means[i] = np.mean(num) / np.mean(den)
    return images, sums, means


def difference(truth, predictions):
    """
    :param truth: ground truth
    :param predictions: predictions of network
    :return: relative error(array)
    """
    images = np.zeros_like(truth)
    sums = np.zeros(predictions.shape[0])  # sample size
    means = np.zeros(predictions.shape[0])
    for i in range(0, predictions.shape[0]):  # loop over samples
        diff = np.abs(predictions[i, 0, :, :] - truth[i, 0, :, :])
        images[i, :, :, :] = diff
        sums[i] = np.sum(diff)
        means[i] = np.mean(diff)
    return images, sums, means


def arg_getter(truth, predictions):
    """
    orders predictions according to their rel. error
    :param truth: ground truth
    :param predictions: output from network
    :return: list of ordered sample indices in decreasing order
    """
    _, test, _ = relative_error(truth, predictions)
    sort = np.asarray(sorted(test))
    sorted_args = [list(test).index(error) for error in sort]
    # decreasing order, arg 0 is the best, -1 is the worst
    return sorted_args

"""
VISUALISATION
"""


def visualise_data(images, cmap='viridis', facecolor='w'):
    """
    Plots random elements e.g. from training dataset.
    :param images: 4D np array, image frames to plot. Dimensions: (#data in set, #frames, h, w)
    :param cmap: colormap
    :param facecolor: background color
    """

    num_img = np.shape(images)[0]
    n = np.random.randint(0, num_img)

    plt.figure(num=None, dpi=80, facecolor=facecolor)
    l = np.shape(images)[1]
    for i in range(l):
        plt.subplot(1, l, i + 1)
        plt.imshow(np.ma.masked_where(images[n][i] < 0, images[n][i]), cmap=cmap)
        plt.title(f"Instance #{n+1} from {num_img}\nFrame: {i}")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)


def error_distribution(truth, predictions, nbins=20, metric="difference"):
    """
    plot relative error dist. of results
    :param truth: ground truth
    :param predictions: predictions of network
    :param metric: difference or relative_error
    :return: nothing (plots relative error distributions)
    """

    if metric == "relative_error":
        error_images, error_vals, error_means = relative_error(truth, predictions)
    elif metric == "difference:":
        error_images, error_vals, error_means = difference(truth, predictions)
    else:
        sys.exit("Metric must be 'difference' or 'relative_error'.")

    plt.hist(error_vals, nbins)
    plt.xlabel('relative error')
    plt.ylabel('count')
    plt.title('mean = ' + str(np.mean(error_vals))[0:5] + ', min = ' + str(np.min(error_vals))[0:5] + ', max = ' + str(
        np.max(error_vals))[0:5])
    plt.yticks(list(set([int(tick) for tick in plt.yticks()[0]])))
    plt.show()
    return error_images, error_vals, error_means


def result_plotter(indices, datasets, task='prediction'):

    if task == 'prediction':
        title = ['Frame t', 'Frame t+1', 'Prediction t+1', 'Relative error']
    elif task == 'upsampling':
        title = ['Original', 'Downsampled', 'Upsampled', 'Relative error']
    else:
        sys.exit("Task must be 'prediction' or 'upsampling'.")
    for i in indices:
        fig, axes = plt.subplots(nrows=1, ncols=4, num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
        for j, ax in enumerate(axes.flat):
            im = ax.imshow(datasets[j][int(i), 0], vmin=0,
                           vmax=max([np.max(dset[int(i)]) for dset in datasets[:2]]) if int(j) < 3 else None,
                           norm=colors.PowerNorm(gamma=0.5) if int(j) == 3 else None)
            ax.set_title(f"{title[j]}", fontsize=10)
            colorbar(im)
            ax.axis('off')
    plt.savefig('foo.png')
    plt.show()


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
