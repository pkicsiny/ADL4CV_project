import os
import numpy as np
import pandas as pd
import sys
import re
import pickle
import io
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from ipywidgets import interact, IntSlider
from IPython.display import display
from sklearn.utils import shuffle
from scipy import signal
from scipy.signal import savgol_filter
from scipy import ndimage


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


def load_datasets(past_frames=1, future_frames=1, prediction=False, steps=1):
    """
    If the data is already saved in .npz file then use this to load it.
    :param past_frames: int, no. of past frames
    :param future_frames: int, no. of predictable frames
    :param prediction: bool, if True it loads a long sequence for sequence prediction testing, else it loads the normal
    dataset for training
    :param steps: int, no. of steps btw. each frames
    :return: train, validation, test sets.
    """
    if not prediction:
        if steps == 1 :
            train = decompress_data(filename=sys.path[0] + "/5min_train_compressed.npz")[
                    :, :, :, :past_frames+future_frames]
            xval = decompress_data(filename=sys.path[0] + "/5min_xval_compressed.npz")[
                   :, :, :, :past_frames+future_frames]
            test = decompress_data(filename=sys.path[0] + "/5min_test_compressed.npz")[
                   :, :, :, :past_frames+future_frames]
        elif steps > 1 :
            idx = range(0, (past_frames+future_frames)*steps, steps)
            if max(idx) >= 8 :
                print('max index is over 8, do not have data.')
                return
            train = decompress_data(filename=sys.path[0] + "/5min_train_compressed.npz")[
                    :, :, :, idx]
            xval = decompress_data(filename=sys.path[0] + "/5min_xval_compressed.npz")[
                   :, :, :, idx]
            test = decompress_data(filename=sys.path[0] + "/5min_test_compressed.npz")[
                   :, :, :, idx]
        print(f"Training data: {train.shape}\nValidation data: {xval.shape}\nTest data: {test.shape}")
        return train, xval, test
    else:
        long_test = decompress_data(filename=sys.path[0] + "/5min_long_pred_compressed.npz")
        print(f"Test data: {long_test.shape}")
        return long_test

def reduce_dims(images):
    """
    :param images: data with 5 dimensions
    :return: same data with 4 dimensions, the images have been split along the t axis and reconcatenated along the
    last (channel) axis: (1000, t, 64, 64, 1) -> (1000, 64, 64, t)
    """
    if len(images.shape) > 4:
        # sequence (t axis)is defined as a separate dim than channel -> make the sequence dim to be the channel dim
        images = np.concatenate([images[:, t, :, :, :] for t in len(images.shape[1])], axis=-1)
    return images


def split_datasets(train, xval, test, past_frames=1, future_frames=1, augment=False, shuffle_training_data=False):
    """
    Further splits data to input and ground truth datasets. If the data has rain and wind velocity maps, they must be
    in the last channel so everything here compiles for them automatically.
    :param train: numpy array of training dataset, shape: (n, w, h, c)
    :param xval: numpy array of validation dataset, shape: (n, w, h, c)
    :param test: numpy array of test dataset, shape: (n, w, h, c)
    :param past_frames: int, no. of consec. frames that will be the input of the network
    :param future_frames: int, no. of consec. frames that will be the ground truth of the network
    :param augment: bool, if True, uses data augmentation by flip and rotation -> 8*(# of data)
    :param shuffle_training_data: bool, shuffle data along first axis or not.
    :return: 6 numpy arrays
    """
    assert past_frames + future_frames <= train.shape[1], "Wrong frame specification!"
    assert past_frames > 0, "No. of past frames must be a positive integer!"
    assert future_frames > 0, "No. of future frames must be a positive integer!"
    if augment:
        print("Data augmentation.")
    x = augment_data(train, shuffle_training_data) if augment else train
    training_data = x[:, :, :, :past_frames]
    trainig_data_truth = x[:, :, :, past_frames:past_frames + future_frames]
    validation_data = xval[:, :, :, :past_frames]
    validation_data_truth = xval[:, :, :, past_frames:past_frames + future_frames]
    test_data = test[:, :, :, :past_frames]
    test_data_truth = test[:, :, :, past_frames:past_frames + future_frames]

    print("Shape of training data: ", training_data.shape, "\nShape of training truth: ", trainig_data_truth.shape,
          "\nShape of validation data: ", validation_data.shape, "\nShape of validation truth: ",
          validation_data_truth.shape,
          "\nShape of test data: ", test_data.shape, "\nShape of test truth: ", test_data_truth.shape)
    return training_data, trainig_data_truth, validation_data, validation_data_truth, test_data, test_data_truth


def get_rain_grid_coords(directory="rain_grid_coordinates"):
    """
    Reads the GPS latitude, longitude values of the rain radar grid into a pandas dataframe. The content
    of the current file are the center coordinates of each grid cell.
    :param directory: folder of the data file
    :return: dataframe of [latitude, longitude, ID] ID is an arbitrary integer enumeration of the cells.
    """
    lon, lat = [pd.DataFrame([re.findall('..\......', row[0]) for idx,
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
    germany = pd.DataFrame(data={'LAT': w_vel['lat'][:][~w_vel['FF'][0].mask],
                                 'LON': w_vel['lon'][:][~w_vel['FF'][0].mask],
                                 'CELL_ID': pd.DataFrame(w_vel['FF'][0].flatten()).dropna().index.values})[[
                                                            'LAT', 'LON', "CELL_ID"]]
    # get closest radar grid cell to each wind cell
    germany["closest_idx"] = -1
    for i, point in enumerate(germany["CELL_ID"]):
        update_output(f"[{i}/{len(germany)}]")
        # dists = germany.iloc[point][["LAT","LON"]] - coords[["LAT","LON"]]
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


def generate_datasets(rain, wind=None, n=10, size=64, length=3, split=None, normalize=False, task="prediction",
                      down_factor=4):
    """
    Use this if you cannot use load_datasets() bc. there are no saved files but you have the raw data imported.
    Splits input array into training, cross validation and test sets.
    :param rain: 3D np array, dataset to be split. This is a channel*height*width large array of rain radar maps.
    Data is created by cutting size*size frames randomly from length consecutive map.
    :param wind: 3D np array, dataset to be split. This is a channel*height*width large array of wind radar maps.
    Data is created by cutting size*size frames randomly from length consecutive map.
    :param n: int, total number of data instances to make
    :param size: int, height and width in pixels of each frame
    :param length: int, length in "time". Defines how many consecutive frames will belong to one instance.
    :param split: list or np array of either floats between 0 and 1 or positive integers. Set to None by deafult
    which means no splitting, just return all instances in one set.
    :param normalize: bool, normalize into [0, 1] or not
    :param task: str, either "prediction" or "upsampling" (for experimenting)
    :param down_factor: int, donwsampling factor if the task is upsampling.
    :return: 3D np array, either one dataset of smaller image frames or three datasets for training,
    cross validating and testing. shape is: (n, h, w, c) If wind is given, shape will be (n, h, w, c, m) where m=3
    is the weather measurable: rho, vx, vy
    """
    ch = np.shape(rain)[0]
    norm_factors = []  # store normalization max values
    if wind is not None:
        # wind timestamps are taken at xx:00:00 but rain is taken at xx:50:00
        # so I take the next index for the wind maps
        ch -= 1
    h = np.shape(rain)[1]
    w = np.shape(rain)[2]

    if task == "upsampling":
        length = 1
    elif task != "prediction":
        sys.exit("Task must be 'prediction' or 'upsampling'.")

    if wind is None:
        images = np.empty((n, size, size, length))  # n series, each consisting of length frames of size size**2
    else:
        images = np.empty((n, size, size, length, 3))  # extra dim for rain and wind maps

    for i in range(n):
        update_output(f"[{i+1}/{n}]")
        # draw 3 random numbers for map number and idx of top left pixel of window
        valid = 0
        if wind is not None:
            image = np.empty((length, size, size, 3))
        else:
            image = np.empty((length, size, size))

        while not valid:
            anchor = (np.random.randint(0, ch - length), np.random.randint(0, h - size), np.random.randint(0, w - size))
            for j in range(length):
                rain_cut = rain[anchor[0] + j]
                if wind is not None:
                    image[j, :, :, 0] = rain_cut[anchor[1]:anchor[1] + size,
                                                 anchor[2]:anchor[2] + size]  # .filled(np.nan)
                else:
                    image[j, :, :] = rain_cut[anchor[1]:anchor[1] + size, anchor[2]:anchor[2] + size]
                    image[j, :, :][image[j, :, :] < 0] = np.nan
            # masked array will get unmasked and nans become huge values
            image[(np.isnan(image)) | (image > 1e5)] = np.nan  # this will be filtered out in the valid method
            if wind is not None:
                valid = valid_image(image[:, :, :, 0])
            else:
                valid = valid_image(image)
        # valid image found, append to final array
        if wind is None:
            norm_factor = np.array(image[image < 1e5]).max()
            norm_factors.append(norm_factor)
            images[i] = np.transpose(image / norm_factor,
                                     (1, 2, 0)) if normalize else np.transpose(image, (1, 2, 0))
        else:
            # get wind data
            for k in range(length):
                vx = -np.flip(np.sin(np.deg2rad(wind[anchor[0] + 1 + k])), axis=0)
                vy = -np.flip(np.cos(np.deg2rad(wind[anchor[0] + 1 + k])), axis=0)
                image[k, :, :, 1] = vx[anchor[1]:anchor[1] + size, anchor[2]:anchor[2] + size].filled(np.nan)
                image[k, :, :, 2] = vy[anchor[1]:anchor[1] + size, anchor[2]:anchor[2] + size].filled(np.nan)
            image[(np.isnan(image)) | (image > 1e5)] = np.nan
            if normalize:
                norm_factor = image[:, :, :, 0][image[:, :, :, 0] < 1e5].max()
                image[:, :, :, 0] = image[:, :, :, 0] / norm_factor
                norm_factors.append(norm_factor)
            # transpose images to have channel as second last dim
            images[i] = np.transpose(image, (1, 2, 0, 3))

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
            low_res_train, low_res_xval, low_res_test = [data[:, ::down_factor, ::down_factor] for data in
                                                         [train, xval, test]]
            update_output(txt + f"\n\nShape of downsampled data:\n\n" +
                          f"Training set: {np.shape(low_res_train)}\n" +
                          f"Validation set: {np.shape(low_res_xval)}\nTest set: {np.shape(low_res_test)}")
            return {"low_res_train": low_res_train,
                    "low_res_xval": low_res_xval,
                    "low_res_test": low_res_test,
                    "train": train,
                    "xval": xval,
                    "test": test,
                    "norm factors": norm_factors}
        else:
            update_output(txt)
            return {"train": train, "xval": xval, "test": test, "norm factors": norm_factors}
    else:  # no split
        if task == "upsampling":  # downsample
            low_res_data = images[:, ::down_factor, ::down_factor]
            update_output(txt + f"\nShape of downsampled data: {np.shape(low_res_data)}")
            return {"low_res_data": low_res_data, "images": images, "norm factors": norm_factors}
        else:
            update_output(txt)
            return {"images": images, "norm factors": norm_factors}


def valid_image(image):
    """
    Filters out some useless data. In the junk variable several conditions are defined to check on the images.
    Currently it checks the number of different entry values and if 0 or 1 makes up 0.75 part of the whole data.
    This throws out the cuts made inside the mask region and rainless areas.
    :param image: 3D np array, dimensions are the number of consecutive frames, height and width
    :return: bool, whether the data instance is valid in terms of usability
    """
    junk = [len(set(np.array(frame).flatten()[
                        np.array(frame).flatten() < 1e5])) <= 8 for frame in image]
    # no nans
    junk += [len(np.array(frame).flatten()[
                     np.array(frame).flatten() < 1e5]) < len(np.array(frame).flatten()) for frame in image]
    # many zeros
    junk += [len(np.array(frame).flatten()[
                     np.array(frame).flatten() == 0]) > 0.8*len(np.array(frame).flatten()) for frame in image]
    # few zeros
    junk += [len(np.array(frame).flatten()[
                     np.array(frame).flatten() == 0]) < 0.5*len(np.array(frame).flatten()) for frame in image]
    return 0 if any(junk) else 1


def preprocess_flow(vx, vy):
    """
    Applies a normalization into [-1, 1] and a median filter smoothing on the flow images.
    :param vx: np array, optical flow x component, shape: (n, h, w, 1)
    :param vy: np array, optical flow y component, shape: (n, h, w, 1)
    :return: preprocessed flows as np array, shape: (n, h, w, 2)
    """
    normalized_optical_flow = normalize_flows(vx, vy)
    median = np.transpose([[
        ndimage.median_filter(
            image[..., ch], 4) for ch in range(2)] for image in normalized_optical_flow], (0, 2, 3, 1))
    return median


def normalize_flows(flow_x, flow_y):
    """
    Normalizes inputs into [-1, 1]
    :param flow_x: np array, optical flow x component, shape: (n, h, w, 1)
    :param flow_y: np array, optical flow y component, shape: (n, h, w, 1)
    :return: normalized flows as np array, shape: (n, h, w, 2)
    """
    flow = np.concatenate((flow_x, flow_y), axis=-1)
    return np.array([
        2*(flow_pair - np.min(flow_pair)) / (np.max(flow_pair) - np.min(flow_pair)) - 1 for flow_pair in flow])


"""
DATA AUGMENTATION
"""


def augment_data(data, shuffle_training_data=False):
    """
    Does data augmentation by rotations in 90, 180 and 270 degrees.
    :param data: np array of shape: (n, h, w, ch) where ch is the "time axis". Input data to be augmented.
    :param shuffle_training_data: bool, if True then it shuffles the data before returning. Otherwise
    the augmented rotations are next to each other in the array.
    :return: Augmented data as np array of shape: (4*n, h, w, ch)
    """
    augmented_data = np.reshape([np.array([
                        data_sample,
                        rotate(data_sample, "90"),
                        rotate(data_sample, "180"),
                        rotate(data_sample, "270")]) for data_sample in data], ((data.shape[0]*4,)+data.shape[1:]))
    if shuffle_training_data:
        return shuffle(augmented_data)
    else:
        return augmented_data


def rotate(img, degree):
    """
    Function to do rotations by "degree" degrees.
    :param img: Image to be rotated. numpy array of shape: (h, w, ch)
    :param degree: int. Degrees to rotate with.
    :return: Rotated image as np array of same shape as input.
    """
    assert degree in ["90", "-270", "180", "-90", "270"], "Rotation degree must be in: [90, 180, 270, -90, -270]"
    rotated = np.rot90(img)
    if degree in ["180", "-90", "270"]:
        rotated = np.rot90(rotated)
    if degree in ["-90", "270"]:
        rotated = np.rot90(rotated)
    return rotated


"""
UTILS
"""


def noisy_d_labels(real, fake):
    """
    idea: https://arxiv.org/pdf/1606.03498.pdf
    Switches the label for 5% of the samples. Also does one sided label smoothing: 0.9 instead of 1.
    :param real: numpy array of labels for the real images. Usually 1.
    :param fake: same for fake images. Usually 0.
    :return: Modified label arrays.
    """
    batch_size = len(real)
    five_percent = int(0.05*batch_size)
    idx = np.random.randint(0, batch_size, five_percent)
    d_real = np.ones_like(real)*0.9
    d_fake = np.zeros_like(fake)
    d_real[idx] = 0.9
    d_fake[idx] = 1
    return d_real, d_fake


def optical_flow(prev, curr, window_size=4, tau=1e-2, init=0):
    """
    Calculates the dense optical flow x and y components between prev and curr.
    Requires import scipy signal and numpy.
    :param prev: numpy array of shape (n, h, w, 1)
    :param curr: numpy array of shape (n, h, w, 1)
    :param window_size: int, kernel window size
    :param tau: float, threshold value for eigenvalues
    :param init: 0 or 1. If 1, prints progress with the update_output() method.
    :return: 2 numpy arrays of shape (n, h, w, 1)
    """
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = int(window_size/2)  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    u = np.zeros(prev.shape)
    v = np.zeros(prev.shape)
    # within window window_size * window_size
    for sample in range(prev.shape[0]):  # loop over samples
        if init == 1:
            update_output(f"[{sample+1}/{prev.shape[0]}]")
        fx = signal.convolve2d(prev[sample, :, :, 0], kernel_x, boundary='symm', mode=mode)
        fy = signal.convolve2d(prev[sample, :, :, 0], kernel_y, boundary='symm', mode=mode)
        ft = signal.convolve2d(curr[sample, :, :, 0], kernel_t, boundary='symm', mode=mode) + signal.convolve2d(
                               prev[sample, :, :, 0], -kernel_t, boundary='symm', mode=mode)
        for i in range(w, int(prev[sample, :, :, 0].shape[0]-w)):
            for j in range(w, int(prev[sample, :, :, 0].shape[1]-w)):
                Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
                b = np.reshape(It, (It.shape[0], 1))
                A = np.vstack((Ix, Iy)).T
                # if threshold tau is larger than the smallest eigenvalue of A'A:
                if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                    nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                    u[sample, i, j, 0] = nu[0]
                    v[sample, i, j, 0] = nu[1]
    print(f"Optical flow map shapes: vx: {u.shape}, vy: {v.shape}")
    return u, v


def advect(image, order=1):  # (64,64,3)
    """
    Applies the physical advection (material derivative) on a density (rain) frame.
    Sources: https://en.wikipedia.org/wiki/Advection
             https://perswww.kuleuven.be/~u0016541/Talks/advection.pdf
             http://mantaflow.com/scenes.html
    :param image: np array of shape (h, w, 3).
     One image with 3 channels: the advected material (rain desity) and the flow field (wind) x and y components.
    :param order: int, 1 or 2. Order of discretization for the advection equation.
    :return: np array of shape (h, w, 1). Advected frame.
    """
    assert order in [1, 2], "This supports only first and second order advection."
    # create array for advected frame
    advected = np.empty_like(image[..., 0:1])
    # pad image
    padded = np.pad(image, (1, 1), 'edge')[:, :, 1:-1]
    # set nans to 0
    padded[np.isnan(padded)] = 0
    if order == 1:
        advected = image[..., 0] - \
             0.5 * image[..., 1] * (padded[2:, :, 0] - padded[:-2, :, 0])[:, 1:-1] - \
             0.5 * image[..., 2] * (padded[:, 2:, 0] - padded[:, :-2, 0])[1:-1]
    elif order == 2:
        advected = image[..., 0] - \
             0.5 * image[..., 1] * (padded[2:, :, 0] - padded[:-2, :, 0])[:, 1:-1] - \
             0.5 * image[..., 2] * (padded[:, 2:, 0] - padded[:, :-2, 0])[1:-1] + \
             0.5 * image[..., 1] ** 2 * (padded[2:, :, 0] + padded[:-2, :, 0] - 2 * padded[1:-1, :, 0])[:, 1:-1] + \
             0.5 * image[..., 2] ** 2 * (padded[:, 2:, 0] + padded[:, :-2, 0] - 2 * padded[:, 1:-1, 0])[1:-1]
    # renormalize (clamp)
    advected[advected < 0] = 0
    advected[advected > 1] = 1
    return advected.reshape((advected.shape + (1,)))  # (64, 64, 1) only density


def freeze_header(df, num_rows=30, num_columns=10, step_rows=1,
                  step_columns=1):
    """
    idea: https://stackoverflow.com/questions/28778668/freeze-header-in-pandas-dataframe
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


def compress_huge_data(huge_data, filename=None):
    """
    Compress large data and SAVE it to a .npz compressed file. Requires io package.
    :param huge_data: file to be compressed
    :param filename: it has to be a string specifying the path and filename
    """
    compressed_array = io.BytesIO()
    np.savez_compressed(filename, compressed_array, huge_data)


def decompress_data(filename=None):
    """"
    Loads and decompresses saved data.
    :param filename: name of saved file
    :return: decompressed data
    """
    return np.load(filename)["arr_1"]


"""
NETWORKS
"""


def unet(input_shape=(64, 64, 1), dropout=0.0, batchnorm=False, kernel_size=4, feature_mult=1, relu_coeff=0.1):
    init = keras.layers.Input(shape=input_shape)

    ConvDown1 = keras.layers.Conv2D(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(init)  # 32
    if batchnorm:
        ConvDown1 = keras.layers.BatchNormalization()(ConvDown1)
    Lr1 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown1)
    if (dropout > 0) and (dropout <= 1):
        Lr1 = keras.layers.Dropout(dropout)(Lr1)

    ConvDown2 = keras.layers.Conv2D(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr1)  # 16
    if batchnorm:
        ConvDown2 = keras.layers.BatchNormalization()(ConvDown2)
    Lr2 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown2)
    if (dropout > 0) and (dropout <= 1):
        Lr2 = keras.layers.Dropout(dropout)(Lr2)

    ConvDown3 = keras.layers.Conv2D(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr2)  # 8
    if batchnorm:
        ConvDown3 = keras.layers.BatchNormalization()(ConvDown3)
    Lr3 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown3)
    if (dropout > 0) and (dropout <= 1):
        Lr3 = keras.layers.Dropout(dropout)(Lr3)

    ConvDown4 = keras.layers.Conv2D(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr3)  # 4
    if batchnorm:
        ConvDown4 = keras.layers.BatchNormalization()(ConvDown4)
    Lr4 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown4)
    if (dropout > 0) and (dropout <= 1):
        Lr4 = keras.layers.Dropout(dropout)(Lr4)

    ConvDown5 = keras.layers.Conv2D(filters=64*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr4)  # 2
    if batchnorm:
        ConvDown5 = keras.layers.BatchNormalization()(ConvDown5)
    Lr5 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown5)
    if (dropout > 0) and (dropout <= 1):
        Lr5 = keras.layers.Dropout(dropout)(Lr5)

    ConvUp4 = keras.layers.Conv2DTranspose(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(Lr5)
    if batchnorm:
        ConvUp4 = keras.layers.BatchNormalization()(ConvUp4)
    Lr6 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp4)
    if (dropout > 0) and (dropout <= 1):
        Lr6 = keras.layers.Dropout(dropout)(Lr6)
    merge1 = keras.layers.concatenate([ConvDown4, Lr6], axis=-1)

    ConvUp3 = keras.layers.Conv2DTranspose(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(merge1)
    if batchnorm:
        ConvUp3 = keras.layers.BatchNormalization()(ConvUp3)
    Lr7 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp3)
    if (dropout > 0) and (dropout <= 1):
        Lr7 = keras.layers.Dropout(dropout)(Lr7)
    merge2 = keras.layers.concatenate([ConvDown3, Lr7], axis=-1)

    ConvUp2 = keras.layers.Conv2DTranspose(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(merge2)
    if batchnorm:
        ConvUp2 = keras.layers.BatchNormalization()(ConvUp2)
    Lr8 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp2)
    if (dropout > 0) and (dropout <= 1):
        Lr8 = keras.layers.Dropout(dropout)(Lr8)
    merge3 = keras.layers.concatenate([ConvDown2, Lr8], axis=-1)

    ConvUp1 = keras.layers.Conv2DTranspose(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(merge3)
    if batchnorm:
        ConvUp1 = keras.layers.BatchNormalization()(ConvUp1)
    Lr9 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp1)
    if (dropout > 0) and (dropout <= 1):
        Lr9 = keras.layers.Dropout(dropout)(Lr9)
    merge4 = keras.layers.concatenate([ConvDown1, Lr9], axis=-1)

    ConvUp0 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                           padding="same", activation='tanh')(merge4)

    return keras.models.Model(inputs=init, outputs=ConvUp0)


def spatial_discriminator(input_shape=(64, 64, 1), condition_shape=(64, 64, 1),
                          dropout=0, batchnorm=False, wgan=False):
    """
    from tempoGAN paper(Appendix A): "BN denotes batch normalization, which is not used in the
    last layer of G, the first layer of Dt and the first layer of Ds [Radford et al. 2016]."
    """
    # condition is the frame t (the original frame) or the sequence of past frames
    condition = keras.layers.Input(shape=condition_shape)
    # other is the generated prediction of frame t+1 or the ground truth frame t+1
    other = keras.layers.Input(shape=input_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = keras.layers.Concatenate(axis=-1)([condition, other])

    conv1 = keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same')(combined_imgs)
    relu1 = keras.layers.LeakyReLU(alpha=0.2)(conv1)
    if (dropout > 0) and (dropout <= 1):
        relu1 = keras.layers.Dropout(dropout)(relu1)

    conv2 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(relu1)
    if batchnorm:
        conv2 = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=0.2)(conv2)
    if (dropout > 0) and (dropout <= 1):
        relu2 = keras.layers.Dropout(dropout)(relu2)

    conv3 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(relu2)
    if batchnorm:
        conv3 = keras.layers.BatchNormalization()(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=0.2)(conv3)
    if (dropout > 0) and (dropout <= 1):
        relu3 = keras.layers.Dropout(dropout)(relu3)

    conv4 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(relu3)
    if batchnorm:
        conv4 = keras.layers.BatchNormalization()(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=0.2)(conv4)
    if (dropout > 0) and (dropout <= 1):
        relu4 = keras.layers.Dropout(dropout)(relu4)

    # Out: 1-dim probability
    flatten = keras.layers.Flatten()(relu4)
    fcl1 = keras.layers.Dense(1)(flatten)
    if not wgan:
        fcl1 = keras.layers.Activation('sigmoid', name="s_disc_output")(fcl1)

    return keras.models.Model(inputs=[condition, other], outputs=fcl1)


def temporal_discriminator(input_shape=(64, 64, 1), advected_shape=(64, 64, 1),
                           dropout=0.3, batchnorm=False, wgan=False):
    # A(G(x_{t-1})) or A(y_{t-1}) (A(frame t)=frame t+1)
    advected = keras.layers.Input(shape=advected_shape)
    # other is the generated prediction of t (frame t+1) or the ground truth of t (frame t+1)
    other = keras.layers.Input(shape=input_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = keras.layers.Concatenate(axis=-1)([advected, other])

    conv1 = keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same')(combined_imgs)
    relu1 = keras.layers.LeakyReLU(alpha=0.2)(conv1)
    if (dropout > 0) and (dropout <= 1):
        relu1 = keras.layers.Dropout(dropout)(relu1)

    conv2 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(relu1)
    if batchnorm:
        conv2 = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=0.2)(conv2)
    if (dropout > 0) and (dropout <= 1):
        relu2 = keras.layers.Dropout(dropout)(relu2)

    conv3 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(relu2)
    if batchnorm:
        conv3 = keras.layers.BatchNormalization()(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=0.2)(conv3)
    if (dropout > 0) and (dropout <= 1):
        relu3 = keras.layers.Dropout(dropout)(relu3)

    conv4 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(relu3)
    if batchnorm:
        conv4 = keras.layers.BatchNormalization()(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=0.2)(conv4)
    if (dropout > 0) and (dropout <= 1):
        relu4 = keras.layers.Dropout(dropout)(relu4)

    # Out: 1-dim probability
    flatten = keras.layers.Flatten()(relu4)
    fcl1 = keras.layers.Dense(1)(flatten)
    if not wgan:
        fcl1 = keras.layers.Activation('sigmoid', name="t_disc_output")(fcl1)

    return keras.models.Model(inputs=[advected, other], outputs=fcl1)


"""
LOSS FUNCTIONS
"""


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    source: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


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

    else:
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


def calculate_skill_scores(ypredicted, ytruth, x=None, threshold=5):
    """
    Calculates some common weather forecasting metrics from these:
    hit: pred=truth=1, miss: pred=0 truth=1, false larm: pred=1 truth=0
    The metrics are: CSI: Critical Success Index, FAR: False Alarm Rate, POD: Probability od Detection
    :param ypredicted: shape (samples,w,h) predictions of network
    :param ytruth: shape (samles,w,h) same as for the predictions, ground truth next frame
    :param x: input frame t (for correlation)
    :param threshold: integer in 0.1mm. Below this means not raining, above this means raining.
    :return: csi, far, pod: lists of length sample_size. Metrics for each image.
    """
    scores = {}

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
    scores["csi"] = hits/(hits+misses+false_alarms)
    # false alarm rate
    scores["far"] = false_alarms/(hits+false_alarms)
    # probability of detection
    scores["pod"] = hits/(hits+misses)

    # correlation
    scores["corr_to_truth"] = [np.sum(
        ypredicted[i]*ytruth[i]) / (np.sqrt(np.sum(ypredicted[i]**2)*np.sum(ytruth[i]**2)) + 1e-9) for i in range(
        len(ypredicted))]
    if x is not None:
        scores["corr_to_input"] = [np.sum(
            ypredicted[i]*x[i]) / (np.sqrt(np.sum(ypredicted[i]**2)*np.sum(x[i]**2)) + 1e-9) for i in range(
            len(ypredicted))]
    return scores


def get_scores(ypred, ytruth, n_next, past, thresholds_as_list=[0.5]):
    """
    Method for calculating evaluation scores.
    :param ypred: np array of predictions. Shape: (n, h, w, t) (t = time axis)
    :param ytruth: np array of ground truth frames. Shape: (n, h, w, n_next) 
    :param n_next: int, how many frames were predicted
    :param past: int, time axis length of input
    :param thresholds_as_list: list of floats, threshold values for binary mapping
    :return: dict of scores where values are a list of n floats; a score number for each pred-truth pair
    """
    scores = {}
    for t in range(n_next): # loop over the predictions
        update_output(t)
        for s in thresholds_as_list: # make a dict entry for each threshold score
            scores[f"pred_{t+1}"] = calculate_skill_scores(ypred[...,t:t+1],
                                                           ytruth[...,past+t:past+1+t],
                                                           x=ytruth[...,past-1:past],
                                                           threshold=s)
    return scores

"""
VISUALISATION
"""


def visualise_data(images, cmap='viridis', facecolor='w'):
    """
    Plots random elements e.g. from training dataset.
    :param images: 4D np array, image frames to plot. Dimensions: (n, h, w, t)
    :param cmap: colormap
    :param facecolor: background color
    """

    num_img = np.shape(images)[0]
    n = np.random.randint(0, num_img)
    t = np.shape(images)[3]

    plt.figure(num=None, dpi=80, facecolor=facecolor)
    for i in range(t):
        if len(np.shape(images)) == 5:  # multi-cannel images: wind, rain
            plt.subplot(3, t, i + 1)
            plt.imshow(images[n,:, :, i, 0], cmap=cmap)
            plt.subplot(3, t, t + i + 1)
            plt.imshow(images[n, :, :, i, 1], cmap=cmap)
            plt.subplot(3, t, 2*t + i + 1)
            plt.imshow(images[n, :, :, i, 2], cmap=cmap)
        else:
            plt.subplot(1, t, i + 1)
            plt.imshow(images[n, :, :, t], cmap=cmap)
        plt.title(f"Instance #{n+1} from {num_img}\nFrame: {i}")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)


def error_distribution(truth, predictions, nbins=20, metric="difference"):
    """
    plot relative error dist. of results
    :param truth: ground truth
    :param predictions: predictions of network
    :param nbins: int, number of bins on the histogram
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


def result_plotter(indices, datasets, save=True):
    """
    Plots result images. Used for generator comparison.
    :param indices: list of integers. These are the indices of the images of the result.
    :param datasets: list of arrays.
    :param save: bool, save figures
    """
    title = ['Frame t', 'Frame t+1', 'Prediction t+1', 'Pixelwise difference']
    for i in indices:
        fig, axes = plt.subplots(nrows=1, ncols=len(datasets), num=None, figsize=(16, 16), dpi=80,
                                 facecolor='w', edgecolor='k')
        for j, ax in enumerate(axes.flat):
            im = ax.imshow(datasets[j][int(i)], vmin=0,
                           vmax=max([np.max(dset[int(i)]) for dset in datasets[:2]]) if int(j) < 3 else None)
            ax.set_title(f"{title[j]}", fontsize=10)
            colorbar(im)
            ax.axis('off')
        if save:
            plt.savefig(f"Plots/Sample_{i}.png")


def colorbar(mappable):
    """
    idea: https://joseph-long.com/writing/colorbars/
    Colorbar fixing function.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def smooth(curve):
    """
    Smoothed curve to plot.
    :param curve: List of floats. The plotted curve to be smoothed.
    :return: Smoothed curve with Savitzky-Golay filter.
    """
    return savgol_filter(curve, 51, 3)


def sequence_prediction_plot(name, test, predictions_dict, past, samples=0):
    """
    Method used for plotting sequence predictions.
    :param name: str, name of the network training. Defined in the beginning of the notebooks.
    :param test: np array, test set for sequence prediction.
    :param predictions_dict: dict, dictionary of predictions. Keys are integers from 0 to the no. of predicted
     consecutive frames.
    :param past: int, number of past frames as input for the generator.
    :param samples: list of integers. The sample indices to be plotted.
    """
    fig, axs = plt.subplots(len(samples)*2,past+len(predictions_dict.keys()), figsize=(32, 32))
    fig.subplots_adjust(wspace=0.3, hspace=0.0)
    for n in range(len(samples)):
        vmax = np.max(test[n, :, :, :past])
        vmin = 0
        for i in range(past):
            im = axs[2*n, i].imshow(test[samples[n], :, :, i], vmax=vmax, vmin=vmin)
            axs[2*n, i].axis('off')
            axs[2*n, i].set_title(f"Past frame {i+1}")
            colorbar(im)
            im = axs[2*n+1, i].imshow(test[samples[n], :, :, i], vmax=vmax, vmin=vmin)
            axs[2*n+1, i].axis('off')
            axs[2*n+1, i].set_title(f"Past frame {i+1}")
            colorbar(im)
        for i in range(past, past+len(predictions_dict.keys())):
            im = axs[2*n, i].imshow(predictions_dict[f"{i-past}"][samples[n], :, :, 0], vmax=vmax, vmin=vmin)
            axs[2*n, i].axis('off')
            axs[2*n, i].set_title(f"Predicted frame {i-past+1}")
            colorbar(im)
            im = axs[2*n+1, i].imshow(test[samples[n], :, :, i], vmax=vmax, vmin=vmin)
            axs[2*n+1, i].axis('off')
            axs[2*n+1, i].set_title(f"Reference frame {i-past+1}")
            colorbar(im)
    fig.savefig(f"Plots/{name}_sequence_prediction.png")
    plt.close()


def validate_on_batch(generator, gan_val, gan_val_truth, batch_size, log, it):
    """
    Validates objective loss on a batch.
    :param generator: keras model for generator
    :param gan_val: np array of validation dataset, shape: (n, h, w, t)
    :param gan_val_truth: np array for validation ground truth, shape: (n, h, w, t)
    :batch_size: int for batch size
    :param log: dict containing lossses as lists
    """
    idx = np.random.choice(gan_val.shape[0], batch_size, replace=False)
    validation_truth = gan_val_truth[idx]
    validation_batch = gan_val[idx]
    validation_loss = generator.test_on_batch(validation_batch, validation_truth)
    log["val_loss"].append(validation_loss)
    log["val_loss_x_coord"].append(it)
    plt.figure()
    plt.plot(log["val_loss_x_coord"], log["val_loss"])
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"Plots/{it}_validation_loss.png")
    plt.close()


def sample_images(epoch, gan_test, gan_test_truth, past_input, generator):
    """
    Samples the first 5 images from the validation or test set during training.
    :param epoch: int, current epoch or iteration
    :param gan_test: np array, validation or test dataset
    :param gan_test_truth: np array, validation or test ground truth dataset
    :param past_input: int, number of past frames in the input
    :param generator: keras model object. The generator of the GAN.
    """
    n = 5
    test_batch = gan_test[:n]
    test_truth = gan_test_truth[:n]
    gen_imgs = generator.predict(test_batch)
    plot_range = past_input
    fig, axs = plt.subplots(n, plot_range+2, figsize=(16, 16))
    for i in range(n):
        vmax = np.max([np.max(test_batch[i]), np.max(test_truth[i])])
        vmin = 0
        for j in range(plot_range):
            im = axs[i, j].imshow(test_batch[i, :, :, j], vmax=vmax, vmin=vmin)
            axs[i, j].axis('off')
            colorbar(im)
            axs[i, j].set_title("Frame t"+str([-past_input+1+j if j < past_input-1 else ""][0]))
        im2 = axs[i, -2].imshow(test_truth[i, :, :, 0], vmax=vmax, vmin=vmin)
        axs[i, -2].axis('off')
        colorbar(im2)
        axs[i, -2].set_title("Frame t+1")
        im3 = axs[i, -1].imshow(gen_imgs[i, :, :, 0], vmax=vmax, vmin=vmin)
        axs[i, -1].axis('off')
        colorbar(im3)
        axs[i, -1].set_title("Prediction t+1")
    fig.savefig("Plots/epoch %d.png" % epoch)
    plt.close()


def plot_training_curves(log, epoch, name, wgan=False):
    """
    Method to plot training curves. Also saves them.
    :param log: dictionary where values are lists of the training losses and metrics.
    :param epoch: int, epoch or iteration number.
    :param name: str, name of the training.
    :param wgan: bool, if True then the plotting algorithm needs to change accordingly.
    """
    total_g_loss = np.array(log["g_loss"])[:, 0]

    total_d_loss = np.array(log["d_loss"])[:, 0] if not wgan else np.array(log["d_loss"])
    smoothed_tgl = smooth(np.array(log["g_loss"])[:, 0])
    smoothed_tdl = smooth(np.array(log["d_loss"])[:, 0]) if not wgan else smooth(np.array(log["d_loss"]))
    objective_loss = np.array(log["g_loss"])[:, 1]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 2]})
    a0.plot(total_g_loss, alpha=0.3, c="b")
    a0.plot(smoothed_tgl, c="b", label="generator")
    a0.grid()
    if wgan:
        a0.plot(np.array(log["d_loss_real"]), c="g", label="real")
        a0.plot(np.array(log["d_loss_fake"]), c="r", label="fake")
    else:
        a0.plot(total_d_loss, alpha=0.3, c="orange")
        a0.plot(smoothed_tdl, c="orange", label="discriminator")
    a0.legend()
    a1.plot(objective_loss, alpha=0.9, c="green", label="L1 objective")
    a1.grid()
    a1.legend()
    f.text(0.5, 0, 'Iterations', ha='center', va='center')
    f.text(0, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

    f.tight_layout()
    f.savefig(f"Plots/{name}_epoch_{epoch}_curves.png")


def plot_temporal_training_curves(log, epoch, name, wgan=False):
    """
    Plot training curves for double discriminator GAN.
    :param log: dict, where the values are lists for the training curves.
    :param epoch: int, number of the current iteration or epoch
    :param name: str, name of the training. Defined in the notebook.
    :param wgan: bool, if True then the plotting algorithm needs to change accordingly.
    """
    total_g_loss = np.array(log["g_loss"])[:, 0]
    smoothed_tgl = smooth(np.array(log["g_loss"])[:, 0])

    objective_loss = np.array(log["g_loss"])[:, 1]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 2]})
    a0.plot(total_g_loss, alpha=0.3, c="b")
    a0.plot(smoothed_tgl, c="b", label="G")
    if wgan:
        a0.plot(np.array(log["ds_loss_real"]), c="g", label="Ds_real")
        a0.plot(np.array(log["ds_loss_fake"]), c="r", label="Ds_fake")
        a0.plot(np.array(log["dt_loss_real"]), c="g", label="Dt_real", linestyle="--")
        a0.plot(np.array(log["dt_loss_fake"]), c="r", label="Dt_fake", linestyle="--")
    else:
        total_ds_loss = np.array(log["ds_loss"])
        total_dt_loss = np.array(log["dt_loss"])
        smoothed_tdsl = smooth(np.array(log["ds_loss"]))
        smoothed_tdtl = smooth(np.array(log["dt_loss"]))
        a0.plot(total_ds_loss, alpha=0.3, c="orange")
        a0.plot(total_dt_loss, alpha=0.3, c="red")
        a0.plot(smoothed_tdsl, c="orange", label="Ds")
        a0.plot(smoothed_tdtl, c="red", label="Dt")
    a0.grid()
    a0.legend()
    a1.plot(objective_loss, alpha=0.9, c="green", label="L1 objective")
    a1.grid()
    a1.legend()
    f.text(0.5, 0, 'Iterations', ha='center', va='center')
    f.text(0, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

    f.tight_layout()
    f.savefig(f"Plots/{name}_epoch_{epoch}_curves.png")


def plot_advections(advected_aux_gen, advected_aux_truth, it):
    """
    Plots some example advected frames during training.
    :param advected_aux_gen: advected frame of a batch of generated images. Shape: (batch_size, h, w, 1)
    :param advected_aux_truth: advected frame of a batch of ground truth images. Shape: (batch_size, h, w, 1)
    :param it: int, current interation number
    """
    gen = advected_aux_gen[:5]
    truth = advected_aux_truth[:5]
    fig, axs = plt.subplots(5, 2, figsize=(16, 16))
    for i in range(5):
        vmax = np.max([np.max(gen[i]), np.max(truth[i])])
        vmin = 0
        im = axs[i, 0].imshow(gen[i, :, :, 0], vmax=vmax, vmin=vmin)
        axs[i, 0].axis('off')
        colorbar(im)
        axs[i, 0].set_title("Advected generated frame")

        im = axs[i, 1].imshow(truth[i, :, :, 0], vmax=vmax, vmin=vmin)
        axs[i, 1].axis('off')
        colorbar(im)
        axs[i, 1].set_title("Advected reference frame")

    fig.savefig("Plots/advections_epoch %d.png" % it)
    plt.close()
