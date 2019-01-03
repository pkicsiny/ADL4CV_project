import os
import numpy as np
import pandas as pd
import sys
import re
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
from matplotlib import colors
import keras.backend as K
from ipywidgets import interact, IntSlider
from IPython.display import display

"""
DATA IMPORT
"""


def get_data(data_dir, which="h", total_length=None, mask=True):
    """
    Imports data from files.
    :param data_dir: string, path of directory containing the data files
    :param which: string, resolution of data: "h", "hourly" for hourly and "5m", "5min" for 5 minutes
    :param total_length: int number of files to read
    :param mask: bool, mask the data with the largest mask among the files
    :return: numpy array imported data
    """
    files = os.listdir(data_dir)
    if total_length is None:
        total_length = len(files)
    # array of inputs
    inputs = np.empty((total_length, 900, 900))
    # import data to the inputs array
    for i, file in enumerate(files):
        if i == total_length:
            break
        update_output(f"[{i+1}/{total_length}]")
        if which in ["h", "hourly"]:
            ascii_grid = np.loadtxt(f"{data_dir}/{files[i]}", skiprows=6)
            inputs[i] = ascii_grid
        elif which in ["5m", "5min", "5minutes", "5minute"]:
            print(data_dir + '/' + file)
            with open(data_dir + '/' + file, "rb") as f:
                byte = f.read()
                start = 0
                for j in range(len(byte)):
                    if byte[j] == 62:
                        start = j
                        break
                inputs[i] = np.flip(np.reshape(np.asarray([c for c in byte[start + 3:]]), (900, 900)), axis=0)
                inputs[i][inputs[i] == 250] = -1

    if mask & (total_length > 100):
        inputs = mask_data(inputs, 100)
    return inputs


def load_datasets(dataset="5min"):
    """
    If the data is already saved in numpy (.npy) file then use this to load it.
    :param dataset: str, which data: hourly or 5 minutes resolution
    :return: train, validation, test sets. Each sample has two channels: time t and t+1
    """
    if dataset in ["5m", "5min", "5minutes", "5minute"]:
        images = np.load(sys.path[0]+"/5_minute.npy").item()
    elif dataset in ["h", "hourly"]:
        images = np.load(sys.path[0]+"/hourly.npy").item()
    train = np.reshape(images["train"], np.shape(images["train"])+(1,))
    xval = np.reshape(images["xval"], np.shape(images["xval"])+(1,))
    test = np.reshape(images["test"], np.shape(images["test"])+(1,))
    print(f"Training data: {train.shape}\nValidation data: {xval.shape}\nTest data: {test.shape}")
    return train, xval, test


def get_rain_grid_coords(directory="rain_grid_coordinates"):
    """
    Reads the GPS latitude, longitude values of the rain radar grid into a pandas dataframe. The content
    of the current file are the center coordinates of each grid cell.
    :param directory: folder of the data file
    :return: dataframe of [latitude, longitude, ID] ID is an arbitrary integer enumeration of the cells.
    """
    lon, lat = [pd.DataFrame([re.findall('..\......',row[0]) for idx,
                              row in pd.read_table(sys.path[0]+f"/{directory}/{file}_center.txt",
                              header=None).iterrows()]) for file in ['lambda', 'phi']]
    coords = pd.DataFrame(columns={"LAT", "LON"})
    coords["LAT"] = np.round(pd.Series([item for sublist in lat.values.tolist() for item in sublist]).astype(float), 4)
    coords["LON"] = np.round(pd.Series([item for sublist in lon.values.tolist() for item in sublist]).astype(float), 4)
    coords["CELL_ID"] = coords.index.values
    return coords


def get_germany(w_vel, coords):
    """
    Fetches GPS coordinates of grid cells inside Germany (a Germany shaped unmasked part of the wind maps).
    Columns are latitude, longitude, cell id (arb. integer, id in original wind grid)
    Then matches nearest neighbor radar cell to each wind (Germany) cell then saves the df in a pickle file.
    So use this if the pickle is not available.
    Works with unflipped (original) wind grid. (Needs flipping around axis 0 afterwards.)
    """
    germany = pd.DataFrame(data={'LAT':w_vel['lat'][:][~w_vel['FF'][0].mask],
                             'LON':w_vel['lon'][:][~w_vel['FF'][0].mask],
                             'CELL_ID':pd.DataFrame(w_vel['FF'][0].flatten()).dropna().index.values})[['LAT', 'LON', "CELL_ID"]]
    #get closest radar grid cell to each wind cell
    germany["closest_idx"] = -1
    for i, point in enumerate(germany["CELL_ID"]):
        update_output(f"[{i}/{len(germany)}]")
        #dists = germany.iloc[point][["LAT","LON"]] - coords[["LAT","LON"]]
        germany["closest_idx"].iloc[point] = np.sqrt((germany.iloc[point]["LAT"] - coords["LAT"])**2 +
                                                     (germany.iloc[point]["LON"] - coords["LON"])**2).idxmin()
    with open('germany.pickle', 'wb') as handle:
        pickle.dump(germany, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
            image[image <= 0] = np.nan() # replace mask values with nan
            valid = valid_image(image)
        images[i] = image

    # normalization each sequence
    if normalize:
        images = np.array([s/s.max() for s in images])

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
                                f"Training set: {np.shape(low_res_train)}\n" +
                                f"Validation set: {np.shape(low_res_xval)}\nTest set: {np.shape(low_res_test)}")
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


def generate_tempoGAN_datasets(rain_density, wind_dir, n=10, length=2, size=64, split=None, normalize=False):
    """
    Splits input array into training, cross validation and test sets.
    :param rain_density: .nc file containing interpolated rain maps on wind grid. A masked np array of shape
     744*938*720.
    :param wind_dir: same as the rain but these are the wind maps on the same grid. They have the same shape. The
    vx and vy channels are calculated from this.
    :param n: int, total number of data instances to make
    :param length: int, number of consecutive frames on time axis to cut. It cuts length frames from all three channels.
    (rho, vx, vy)
    :param size: int, height and width in pixels of each frame
    :param split: list or np array of either floats between 0 and 1 or positive integers. Set to None by deafult
    which means no splitting, just return all instances in one set.
    :param normalize: boolean, if true, the rain maps will be normalized between [0,1]. They are simply divided by the
    max pixel value in the series. (Wind is always between [-1,1])
    :return: 3D np array, either one dataset of smaller image frames or three datasets for training,
    cross validating and testing
    """
    time = 743
    h = rain_density[0].shape[0]  # 938
    w = rain_density[0].shape[1]  # 720

    images = np.zeros(
        (n, length, size, size, 3))  # n series, each of size size**2 and rho,vx,vy,future frames as channels
    for i in range(n):
        update_output(f"[{i+1}/{n}]")
        # draw 3 random numbers for map number and idx of top left pixel of window
        valid = 0
        while not valid:
            anchor = (np.random.randint(0, time - 2), np.random.randint(0, h - size), np.random.randint(0, w - size))
            image = np.empty((length, size, size, 3))
            for j in range(length):
                r = rain_density[anchor[0] + j]
                x = -np.flip(np.sin(np.deg2rad(wind_dir[anchor[0] + 1 + j])), axis=0)
                y = -np.flip(np.cos(np.deg2rad(wind_dir[anchor[0] + 1 + j])), axis=0)
                image[j, :, :, 0] = r[anchor[1]:anchor[1] + size, anchor[2]:anchor[2] + size].filled(np.nan)
                image[j, :, :, 1] = x[anchor[1]:anchor[1] + size, anchor[2]:anchor[2] + size].filled(np.nan)
                image[j, :, :, 2] = y[anchor[1]:anchor[1] + size, anchor[2]:anchor[2] + size].filled(np.nan)
            # first channel is the current frame, the next two are the wind x and y components where the index is shifted by 1
            # because the rain dates are in xx:50 resolution but the wind is in xx:00. The next channels are the next rain fields
            # to be predicted
            valid = valid_image(image)
        images[i] += image

    if normalize:  # to [0,1] only for rain
        images[:, :, :, :, 0] = np.array([s[:, :, :, 0] / s[:, :, :, 0].max() for s in images])
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
        update_output(txt)
        return {"train": train,
                "xval": xval,
                "test": test, }
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
    if len(np.shape(image)) == 3:  # one channel frames
        junk = [len(set(np.array(frame).flatten()[
                            ~np.isnan(np.array(frame).flatten())])) <= 8 for frame in image]
    else:  # three channel frames
        # frame is a triplet for tempogan images
        # only rain channel
        junk = [len(set(np.array(frame[:, :, 0]).flatten()[
                            ~np.isnan(np.array(frame[:, :, 0]).flatten())])) <= 8 for frame in image]

    junk += [len(np.array(frame).flatten()[
                     np.isnan(np.array(frame).flatten())]) > 0.25 * len(np.array(frame).flatten()) for frame in image]
    return 0 if any(junk) else 1

"""
UTILS
"""


def freeze_header(df, num_rows=30, num_columns=10, step_rows=1,
                  step_columns=1):
    """
    Freeze the headers (column and index names) of a Pandas DataFrame. A widget
    enables to slide through the rows and columns.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns

    Returns
    -------
    Displays the DataFrame with the widget
    """
    @interact(last_row=IntSlider(min=min(num_rows, df.shape[0]),
                                 max=df.shape[0],
                                 step=step_rows,
                                 description='rows',
                                 readout=False,
                                 disabled=False,
                                 continuous_update=True,
                                 orientation='horizontal',
                                 slider_color='purple'),
              last_column=IntSlider(min=min(num_columns, df.shape[1]),
                                    max=df.shape[1],
                                    step=step_columns,
                                    description='columns',
                                    readout=False,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    slider_color='purple'))
    def _freeze_header(last_row, last_column):
        display(df.iloc[max(0, last_row-num_rows):last_row,
                        max(0, last_column-num_columns):last_column])


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


def custom_loss(loss="l2+gd"):
    def loss_function(y_true, y_pred):
        """
        Method for custom loss. Add new losses as a new if.
        """
        losses = 0
        loss_name = ""
        if type(loss) == str:
            loss_ = pd.Series(list(filter(None, list(re.split("[\+, ]", loss)))))
        else:
            loss_ = pd.Series(loss)

        if loss_.isin(["l1", "L1", "l_1", "L_1", "mae", "MAE", "mean_absolute_error"]).any():
            losses += keras.losses.mean_absolute_error(y_true, y_pred)
            loss_name += "L1 + "
        if loss_.isin(["l2", "L2", "l_2", "L_2", "mse", "MSE", "mean_squared_error", "mean_square_error"]).any():
            losses += keras.losses.mean_squared_error(y_true, y_pred)
            loss_name += "L2 + "
        if loss_.isin(["gd", "gdl", "gradient", "gradient_loss", "gradient_diff", "gradient_difference",
                       "gradient_difference_loss"]).any():
            losses += gradient_diff(y_true, y_pred)
            loss_name += "gradient difference + "

        if losses != 0:
            print(f"***Using {loss_name[:-2]}loss.***")
        else:
            print("***Loss not understood. Using L2 loss as default.***")
            losses += keras.losses.mean_squared_error(y_true, y_pred)

        return losses

    return loss_function


def gradient_diff(yTrue, yPred):
    """
    Channels last. Images can be either 4 or 5 dim depending on the data shape needed for the networks.
    H and W are dims 2,3 for 5D and 1,2 for 4D data.
    :param yTrue: ground truth
    :param yPred: prediction of network
    :return: gradient loss from https://arxiv.org/pdf/1511.05440.pdf
    """
    alpha = 1
    if len(yTrue.shape) == len(yPred.shape) == 4:
        true = K.pow(K.flatten(K.abs(K.abs(yTrue[:, 1:, :, :] - yTrue[:, :-1, :, :]) -
                               K.abs(yPred[:, 1:, :, :] - yPred[:, :-1, :, :]))), alpha)
        pred = K.pow(K.flatten(K.abs(K.abs(yTrue[:, :, 1:, :] - yTrue[:, :, :-1, :]) -
                               K.abs(yPred[:, :, 1:, :] - yPred[:, :, :-1, :]))), alpha)

    elif len(yTrue.shape) == len(yPred.shape) == 5:
        true = K.pow(K.flatten(K.abs(K.abs(yTrue[:, :, 1:, :, :] - yTrue[:, :, :-1, :, :]) -
                               K.abs(yPred[:, :, 1:, :, :] - yPred[:, :, :-1, :, :]))), alpha)
        pred = K.pow(K.flatten(K.abs(K.abs(yTrue[:, :, :, 1:, :] - yTrue[:, :, :, :-1, :]) -
                               K.abs(yPred[:, :, :, 1:, :] - yPred[:, :, :, :-1, :]))), alpha)
    num = K.sum(true + pred)
    return num / tf.to_float((K.shape(true)[0] + K.shape(pred)[0]))


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
        num = np.abs(predictions[i] - truth[i])
        den = np.abs(truth[i])
        images[i] = np.divide(num, den)
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
        diff = np.abs(predictions[i] - truth[i])
        images[i] = diff
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
    sorted_args = [list(test).index(e) for e in sort]
    # decreasing order, arg 0 is the best, -1 is the worst
    return sorted_args


def calculate_skill_scores(ypredicted, ytruth, threshold=5):
    """
    Calculates some common weather forecasting metrics from these:
    hit: pred=truth=1, miss: pred=0 truth=1, false larm: pred=1 truth=0
    The metrics are: CSI: Critical Success Index, FAR: False Alarm Rate, POD: Probability od Detection
    :param ypredicted: shape (samples,w,h) predictions of network
    :param ytruth: shape (samles,w,h) same as for the predictions, ground truth next frame
    :param threshold: integer in 0.1mm. Below this means not raining, above this means raining.
    :return: csi, far, pod: lists of length sample_size. Metrics for each image.
    """
    binary_pred = np.zeros_like(ypredicted)
    binary_truth = np.zeros_like(ytruth)
    binary_pred[ypredicted >= threshold] = 1
    binary_truth[ytruth >= threshold] = 1
    # hits where the truth and prediction pixel falls to the same side of the threshold (both are 1)
    # (e.g. rains on truth and rains on prediction)
    # shape is (no. samples), each element is an int of matching pixels
    # pred = 1 and truth = 1
    hits = np.array([len(np.intersect1d(np.where(binary_pred[i].flatten()),
                         np.where(binary_truth[i].flatten()))) for i in range(len(binary_truth))])
    # pred = 0 and truth = 1
    misses = np.array([len(np.intersect1d(np.where(binary_pred[i].flatten() == 0),
                           np.where(binary_truth[i].flatten()))) for i in range(len(binary_truth))])
    # pred = 1 and truth = 0
    false_alarms = np.array([len(np.intersect1d(np.where(binary_pred[i].flatten()),
                                 np.where(binary_truth[i].flatten() == 0))) for i in range(len(binary_truth))])
    # critical success index
    csi = hits/(hits+misses+false_alarms)
    # false alarm rate
    far = false_alarms/(hits+false_alarms)
    # probability of detection
    pod = hits/(hits+misses)
    return csi, far, pod

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
    elif metric == "difference":
        error_images, error_vals, error_means = difference(truth, predictions)
    else:
        sys.exit("Metric must be 'difference' or 'relative_error'.")

    plt.hist(error_vals, nbins)
    plt.xlabel(f"{metric}")
    plt.ylabel('count')
    plt.title('mean = ' + str(np.mean(error_vals))[0:5] + ', min = ' + str(np.min(error_vals))[0:5] + ', max = ' + str(
        np.max(error_vals))[0:5])
    plt.yticks(list(set([int(tick) for tick in plt.yticks()[0]])))
    plt.savefig(f"Plots/{metric}.png")
    plt.show()
    return error_images, error_vals, error_means


def result_plotter(indices, datasets):
    """
    Plots result images.
    :param indices: list of integers. These are the indices of the images of the result.
    :param datasets: list of arrays.
    """
    title = ['Frame t', 'Frame t+1', 'Prediction t+1', 'Pixelwise difference']
    for i in indices:
        fig, axes = plt.subplots(nrows=1, ncols=4, num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
        for j, ax in enumerate(axes.flat):
            im = ax.imshow(datasets[j][int(i)], vmin=0,
                           vmax=max([np.max(dset[int(i)]) for dset in datasets[:2]]) if int(j) < 3 else None)
            # , norm=colors.PowerNorm(gamma=0.5) if int(j) == 3 else None)
            ax.set_title(f"{title[j]}", fontsize=10)
            colorbar(im)
            ax.axis('off')
        plt.savefig(f"Plots/Sample_{i}.png")
    plt.show()


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
