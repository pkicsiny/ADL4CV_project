import src
import numpy as np

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


def calculate_skill_scores(ypredicted, ytruth, x=None, threshold=0.5):
    """
    Calculates some common weather forecasting metrics from these:
    hit: pred=truth=1, miss: pred=0 truth=1, false larm: pred=1 truth=0
    The metrics are: CSI: Critical Success Index, FAR: False Alarm Rate, POD: Probability od Detection
    :param ypredicted: shape (samples,w,h) predictions of network
    :param ytruth: shape (samles,w,h) same as for the predictions, ground truth next frame
    :param x: input frame t (for correlation)
    :param threshold: float in 0.1mm. Below this means not raining, above this means raining.
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
    for t in range(n_next):  # loop over the predictions
        src.update_output(t)
        for s in thresholds_as_list:  # make a dict entry for each threshold score
            scores[f"pred_{t+1}"] = calculate_skill_scores(ypred[..., t:t+1],
                                                           ytruth[..., past+t:past+1+t],
                                                           x=ytruth[..., past-1:past],
                                                           threshold=s)
    return scores
