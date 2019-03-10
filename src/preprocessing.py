import src
import numpy as np
import sys
from scipy import ndimage


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
        src.update_output(f"[{i+1}/{n}]")
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
            src.update_output(txt + f"\n\nShape of downsampled data:\n\n" +
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
            src.update_output(txt)
            return {"train": train, "xval": xval, "test": test, "norm factors": norm_factors}
    else:  # no split
        if task == "upsampling":  # downsample
            low_res_data = images[:, ::down_factor, ::down_factor]
            src.update_output(txt + f"\nShape of downsampled data: {np.shape(low_res_data)}")
            return {"low_res_data": low_res_data, "images": images, "norm factors": norm_factors}
        else:
            src.update_output(txt)
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
                     np.array(frame).flatten() == 0]) > np.floor(0.8*len(np.array(frame).flatten())) for frame in image]
    # few zeros
    junk += [len(np.array(frame).flatten()[
                     np.array(frame).flatten() == 0]) < np.floor(0.5*len(np.array(frame).flatten())) for frame in image]
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
