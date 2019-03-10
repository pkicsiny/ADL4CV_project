import src
import sys
import os
import numpy as np
import re
import pandas as pd
import pickle


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
        src.update_output(f"[{i+1}/{total_length}]")
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
        inputs = src.mask_data(inputs, 100)
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
        if steps == 1:
            train = src.decompress_data(filename=sys.path[0] + "/5min_train_compressed.npz")[
                    :, :, :, :past_frames+future_frames]
            xval = src.decompress_data(filename=sys.path[0] + "/5min_xval_compressed.npz")[
                   :, :, :, :past_frames+future_frames]
            test = src.decompress_data(filename=sys.path[0] + "/5min_test_compressed.npz")[
                   :, :, :, :past_frames+future_frames]
        else:  # steps > 1:
            idx = range(0, (past_frames+future_frames)*steps, steps)
            if max(idx) >= 8:
                print('max index is over 8, do not have data.')
                return
            train = src.decompress_data(filename=sys.path[0] + "/5min_train_compressed.npz")[
                    :, :, :, idx]
            xval = src.decompress_data(filename=sys.path[0] + "/5min_xval_compressed.npz")[
                   :, :, :, idx]
            test = src.decompress_data(filename=sys.path[0] + "/5min_test_compressed.npz")[
                   :, :, :, idx]
        print(f"Training data: {train.shape}\nValidation data: {xval.shape}\nTest data: {test.shape}")
        return train, xval, test
    else:
        long_test = src.decompress_data(filename=sys.path[0] + "/5min_long_pred_compressed.npz")
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
        images = np.concatenate([images[:, t, :, :, :] for t in range(0, len(images.shape[1]))], axis=-1)
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
    x = src.augment_data(train, shuffle_training_data) if augment else train
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
        src.update_output(f"[{i}/{len(germany)}]")
        # dists = germany.iloc[point][["LAT","LON"]] - coords[["LAT","LON"]]
        germany["closest_idx"].iloc[point] = np.sqrt((germany.iloc[point]["LAT"] - coords["LAT"])**2 +
                                                     (germany.iloc[point]["LON"] - coords["LON"])**2).idxmin()
    with open('germany.pickle', 'wb') as handle:
        pickle.dump(germany, handle, protocol=pickle.HIGHEST_PROTOCOL)
