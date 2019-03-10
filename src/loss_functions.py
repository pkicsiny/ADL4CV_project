import src
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import pandas as pd
import re

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
