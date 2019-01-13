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


def load_datasets(dataset="5min", past_frames=1, future_frames=1):
    """
    If the data is already saved in .npz file then use this to load it.
    :param dataset: str, which data: hourly or 5 minutes resolution or data for the dual discriminator gan (hourly too)
    :param past_frames: int, no. of past frames
    :param future_frames: int, no. of predictable frames
    :return: train, validation, test sets.
    """
    if dataset in ["temporal", "tempogan", "tampoGAN", "gan", "GAN"]:
        train = decompress_data(filename=sys.path[0] + "/tempoGAN_train_compressed.npz")[:,:,:,:past_frames+future_frames]
        xval = decompress_data(filename=sys.path[0] + "/tempoGAN_xval_compressed.npz")[:,:,:,:past_frames+future_frames]
        test = decompress_data(filename=sys.path[0] + "/tempoGAN_test_compressed.npz")[:,:,:,:past_frames+future_frames]
    if dataset in ["5m", "5min", "5minutes", "5minute"]:
            train = decompress_data(filename=sys.path[0] + "/5min_train_compressed.npz")[:,:,:,:past_frames+future_frames]
            xval = decompress_data(filename=sys.path[0] + "/5min_xval_compressed.npz")[:,:,:,:past_frames+future_frames]
            test = decompress_data(filename=sys.path[0] + "/5min_test_compressed.npz")[:,:,:,:past_frames+future_frames]
    print(f"Training data: {train.shape}\nValidation data: {xval.shape}\nTest data: {test.shape}")
    return train, xval, test


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


def split_datasets(train, xval, test, past_frames=1, future_frames=1, augment=False):
    """
    Further splits data to input and ground truth datasets. If the data has rain and wind velocity maps, they must be
    in the last channel so everything here compiles for them automatically.
    :param train: numpy array of training dataset, shape: (n, w, h, c)
    :param xval: numpy array of validation dataset, shape: (n, w, h, c)
    :param test: numpy array of test dataset, shape: (n, w, h, c)
    :param past_frames: int, no. of consec. frames that will be the input of the network
    :param future_frames: int, no. of consec. frames that will be the ground truth of the network
    :param augment: bool, if True, uses data augmentation by flip and rotation -> 8*(# of data)
    :return: 6 numpy arrays
    """
    assert past_frames + future_frames <= train.shape[1], "Wrong frame specification!"
    assert past_frames > 0, "No. of past frames must be a positive integer!"
    assert future_frames > 0, "No. of future frames must be a positive integer!"
    if augment:
        print("Data augmentation.")
    training_data = augment_data(train[:, :, :, :past_frames]) if augment else train[:, :, :, :past_frames]
    trainig_data_truth = augment_data(train[:, :, :, past_frames:past_frames + future_frames]) if augment else train[
                                                                    :, :, :, past_frames:past_frames + future_frames]
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
    :return: 3D np array, either one dataset of smaller image frames or three datasets for training,
    cross validating and testing. shape is: (n, h, w, c) If wind is given, shape will be (n, h, w, c, m) where m=3
    is the weather measurable: rho, vx, vy
    """
    ch = np.shape(rain)[0]
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
            image[(np.isnan(image)) | (image > 1e5)] = np.nan
            if wind is not None:
                valid = valid_image(image[:, :, :, 0])
            else:
                valid = valid_image(image)
        # valid image found, append to final array
        if wind is None:
            images[i] = np.transpose(image / np.array(image[image < 1e5]).max(),
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
                image[:, :, :, 0] = image[:, :, :, 0] / image[:, :, :, 0][image[:, :, :, 0] < 1e5].max()
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
                    "test": test}
        else:
            update_output(txt)
            return {"train": train, "xval": xval, "test": test}
    else:  # no split
        if task == "upsampling":  # downsample
            low_res_data = images[:, ::down_factor, ::down_factor]
            update_output(txt + f"\nShape of downsampled data: {np.shape(low_res_data)}")
            return {"low_res_data": low_res_data, "images": images}  # data and ground truth
        else:
            update_output(txt)
            return {"images": images}


def valid_image(image):
    """
    Filters out some useless data. In the junk variable several conditions are defined to check on the images.
    Currently it checks the number of different entry values and if 0s or 1s make up 0.75 part of the whole data.
    This throws out the cuts made inside or almost inside the mask region and rainless areas.
    Still can be improved.
    :param image: 3D np array, dimensions are the number of consecutive frames, height and width
    :return: bool, whether the data instance is valid in terms of usability
    """
    junk = [len(set(np.array(frame).flatten()[
                        np.array(frame).flatten() < 1e5])) <= 8 for frame in image]
    junk += [len(np.array(frame).flatten()[
                     np.array(frame).flatten() < 1e5]) < 0.75 * len(np.array(frame).flatten()) for frame in image]
    return 0 if any(junk) else 1

"""
DATA AUGMENTATION
"""

def augment_data(data):
    #dimensions are n, h, w, c
    return np.reshape([np.array([
        data_sample,
        rotate(data_sample, "90"),
        rotate(data_sample, "180"),
        rotate(data_sample, "270"),
        np.flip(data_sample, axis=1),
        np.flip(rotate(data_sample, "90"), axis=1),
        np.flip(rotate(data_sample, "180"), axis=1),
        np.flip(rotate(data_sample, "270"), axis=1)]) for data_sample in data], ((data.shape[0]*8,)+data.shape[1:]))

def rotate(img, degree):

    assert degree in ["90","-270", "180", "-90", "270"], "Rotation degree must be in: [90, 180, 270, -90, -270]"
    rotated = np.rot90(img)
    if degree in ["180", "-90", "270"]:
        rotated = np.rot90(rotated)
    if degree in ["-90", "270"]:
        rotated = np.rot90(rotated)
    return rotated

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


def compress_huge_data(huge_data, filename=None):
    """
    Compress large data and SAVE it to a .npz compressed file. Requires io package.
    :param huge_data: file to be compressed
    :param filename: it has to be a string specifying the path and filename
    """
    compressed_array = io.BytesIO()
    np.savez_compressed(filename,compressed_array, huge_data)


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
# "BN denotes batch normalization, which is not used in the
# last layer of G, the first layer of Dt and the first layer of Ds [Radford et al. 2016]." from tempoGAN paper

def unet(input_shape=(64, 64, 1)):
    init = keras.layers.Input(shape=input_shape)
    ConvDown1 = keras.layers.Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), padding="same")(init)
    Bn1 = keras.layers.BatchNormalization()(ConvDown1)
    Lr1 = keras.layers.LeakyReLU(alpha=0.1)(Bn1)
    # 64
    ConvDown2 = keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding="same")(Lr1)
    Bn2 = keras.layers.BatchNormalization()(ConvDown2)
    Lr2 = keras.layers.LeakyReLU(alpha=0.1)(Bn2)
    # 32
    ConvDown3 = keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="same")(Lr2)
    Bn3 = keras.layers.BatchNormalization()(ConvDown3)
    Lr3 = keras.layers.LeakyReLU(alpha=0.1)(Bn3)
    # 16
    ConvDown4 = keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="same")(Lr3)
    Bn4 = keras.layers.BatchNormalization()(ConvDown4)
    Lr4 = keras.layers.LeakyReLU(alpha=0.1)(Bn4)
    # 8
    ConvDown5 = keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="same")(Lr4)
    Bn5 = keras.layers.BatchNormalization()(ConvDown5)
    Lr5 = keras.layers.LeakyReLU(alpha=0.1)(Bn5)
    # 4

    # 8
    UpSamp1 = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(Lr5)
    merge1 = keras.layers.concatenate([ConvDown4, UpSamp1], axis=-1)
    Conv1 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding="same")(merge1)
    Bn6 = keras.layers.BatchNormalization()(Conv1)
    Lr6 = keras.layers.LeakyReLU(alpha=0.1)(Bn6)
    # 16
    UpSamp2 = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(Lr6)
    merge2 = keras.layers.concatenate([ConvDown3, UpSamp2], axis=-1)
    Conv2 = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding="same")(merge2)
    Bn7 = keras.layers.BatchNormalization()(Conv2)
    Lr7 = keras.layers.LeakyReLU(alpha=0.1)(Bn7)
    # 32
    UpSamp3 = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(Lr7)
    Conv3 = keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), padding="same")(UpSamp3)
    Bn8 = keras.layers.BatchNormalization()(Conv3)
    Lr8 = keras.layers.LeakyReLU(alpha=0.1)(Bn8)
    # 64
    UpSamp4 = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(Lr8)
    Conv4 = keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding="same")(UpSamp4)
    Bn9 = keras.layers.BatchNormalization()(Conv4)
    Lr9 = keras.layers.LeakyReLU(alpha=0.1)(Bn9)

    Conv5 = keras.layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1),
                                padding="same", activation='tanh')(Lr9)

    return keras.models.Model(inputs=init, outputs=Conv5)


def spatial_discriminator(input_shape=(64, 64, 1), condition_shape=(64, 64, 1)):
    # condition is the frame t (the original frame) or the sequence of past frames
    condition = keras.layers.Input(shape=condition_shape)
    # other is the generated prediction of frame t+1 or the ground truth frame t+1
    other = keras.layers.Input(shape=input_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = keras.layers.Concatenate(axis=-1)([condition, other])

    conv1 = keras.layers.Conv2D(filters=4, kernel_size=4, strides=2, padding='same')(combined_imgs)
    Bn1   = keras.layers.BatchNormalization()(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=0.2)(Bn1)

    conv2 = keras.layers.Conv2D(filters=8, kernel_size=4, strides=2, padding='same')(relu1)
    Bn2   = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=0.2)(Bn2)

    conv3 = keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same')(relu2)
    Bn3   = keras.layers.BatchNormalization()(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=0.2)(Bn3)

    conv4 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(relu3)
    Bn4   = keras.layers.BatchNormalization()(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=0.2)(Bn4)

    # Out: 1-dim probability
    flatten = keras.layers.Flatten()(relu4)
    fcl1 = keras.layers.Dense(1)(flatten)
    sig1 = keras.layers.Activation('sigmoid', name="s_disc_output")(fcl1)

    return keras.models.Model(inputs=[condition, other], outputs=sig1)


def temporal_discriminator(input_shape=(64, 64, 1), advected_shape=(64, 64, 1)):
  #  dropout = 0.5
    # A(G(x_{t-1})) or A(y_{t-1}) (A(frame t)=frame t+1)
    advected = keras.layers.Input(shape=advected_shape)
    # other is the generated prediction of t (frame t+1) or the ground truth of t (frame t+1)
    other = keras.layers.Input(shape=input_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = keras.layers.Concatenate(axis=-1)([advected, other])

    conv1 = keras.layers.Conv2D(filters=4, kernel_size=4, strides=2, padding='same')(combined_imgs)
    Bn1   = keras.layers.BatchNormalization()(conv1)
    relu1 = keras.layers.LeakyReLU(alpha=0.2)(Bn1)

    conv2 = keras.layers.Conv2D(filters=8, kernel_size=4, strides=2, padding='same')(relu1)
    Bn2   = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=0.2)(Bn2)

    conv3 = keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same')(relu2)
    Bn3   = keras.layers.BatchNormalization()(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=0.2)(Bn3)

    conv4 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(relu3)
    Bn4   = keras.layers.BatchNormalization()(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=0.2)(Bn4)

    # Out: 1-dim probability
    flatten = keras.layers.Flatten()(relu4)
    fcl1 = keras.layers.Dense(1)(flatten)
    sig1 = keras.layers.Activation('sigmoid', name="t_disc_output")(fcl1)

    return keras.models.Model(inputs=[advected, other], outputs=sig1)


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
            plt.imshow(images[n,:,:,t], cmap=cmap)
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


def result_plotter(indices, datasets, save=True):
    """
    Plots result images.
    :param indices: list of integers. These are the indices of the images of the result.
    :param datasets: list of arrays.
    :param save: bool, save figures
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
        if save:
            plt.savefig(f"Plots/Sample_{i}.png")
    plt.show()


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
