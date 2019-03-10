import src
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter


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
            plt.imshow(images[n, :, :, i, 0], cmap=cmap)
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
        error_images, error_vals, error_means = src.relative_error(truth, predictions)
    elif metric == "difference":
        error_images, error_vals, error_means = src.difference(truth, predictions)
    else:
        sys.exit("Metric must be 'difference' or 'relative_error'.")

    plt.hist(error_vals, nbins)
    plt.xlabel(f"{metric}")
    plt.ylabel('count')
    plt.title('mean = ' + str(np.mean(error_vals))[0:5] + ', min = ' + str(np.min(error_vals))[0:5] + ', max = ' + str(
        np.max(error_vals))[0:5])
    plt.yticks(list(set([int(tick) for tick in plt.yticks()[0]])))
    plt.savefig(f"C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/{metric}.png")
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
            # , norm=colors.PowerNorm(gamma=0.5) if int(j) == 3 else None)
            ax.set_title(f"{title[j]}", fontsize=10)
            colorbar(im)
            ax.axis('off')
        if save:
            plt.savefig(f"C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/Sample_{i}.png")


def colorbar(mappable):
    """
    idea: https://joseph-long.com/writing/colorbars/
    Colorbar fixing function.
    """
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


def sequence_prediction_plot(name, test, predictions_dict, past, samples):
    """
    Method used for plotting sequence predictions.
    :param name: str, name of the network training. Defined in the beginning of the notebooks.
    :param test: np array, test set for sequence prediction.
    :param predictions_dict: dict, dictionary of predictions. Keys are integers from 0 to the no. of predicted
     consecutive frames.
    :param past: int, number of past frames as input for the generator.
    :param samples: list of integers. The sample indices to be plotted.
    """
    fig, axs = plt.subplots(len(samples)*2, past+len(predictions_dict.keys()), figsize=(32, 32))
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
    fig.savefig(f"C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/{name}_sequence_prediction.png")
    plt.close()


def validate_on_batch(generator, gan_val, gan_val_truth, batch_size, log, it):
    """
    Validates objective loss on a batch.
    :param generator: keras model for generator
    :param gan_val: np array of validation dataset, shape: (n, h, w, t)
    :param gan_val_truth: np array for validation ground truth, shape: (n, h, w, t)
    :param batch_size: int for batch size
    :param log: dict containing losses as lists
    :param it: int for iteration number
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
    plt.savefig(f"C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/{it}_validation_loss.png")
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
    fig.savefig("C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/epoch %d.png" % epoch)
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
    f.savefig(f"C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/{name}_epoch_{epoch}_curves.png")


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
    f.savefig(f"C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/{name}_epoch_{epoch}_curves.png")


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

    fig.savefig("C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/ADL4CV_project/Plots/advections_epoch %d.png" % it)
    plt.close()
