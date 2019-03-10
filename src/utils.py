import src
import numpy as np
from scipy import signal
from ipywidgets import interact, IntSlider
from IPython.display import display
from IPython.display import clear_output
import io


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
            src.update_output(f"[{sample+1}/{prev.shape[0]}]")
        fx = signal.convolve2d(prev[sample, :, :, 0], kernel_x, boundary='symm', mode=mode)
        fy = signal.convolve2d(prev[sample, :, :, 0], kernel_y, boundary='symm', mode=mode)
        ft = signal.convolve2d(curr[sample, :, :, 0], kernel_t, boundary='symm', mode=mode) + signal.convolve2d(
                               prev[sample, :, :, 0], -kernel_t, boundary='symm', mode=mode)
        for i in range(w, int(prev[sample, :, :, 0].shape[0]-w)):
            for j in range(w, int(prev[sample, :, :, 0].shape[1]-w)):
                ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                it = ft[i-w:i+w+1, j-w:j+w+1].flatten()
                b = np.reshape(it, (it.shape[0], 1))
                A = np.vstack((ix, iy)).T
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