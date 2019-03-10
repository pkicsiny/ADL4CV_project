import src
import numpy as np
from sklearn.utils import shuffle


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

