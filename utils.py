import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def convert(item):
    if type(item) != np.ndarray:
        item = np.array(item)
    return item


def plot_loss_lr(y, x, plt, label, c):
    """
    Use:
        plt.figure()
        plt = plot_loss(loss, label, plt)
        ...
        plt.show()
    """
    y = convert(y)
    if x is None:
        x = [i for i in range(1, len(y) + 1)]
    x = convert(x)

    plt.plot(x, y, label=label, c=c)

    return plt


def plot_f1_scores(f1, plt, label, c):
    f1_x = np.array(f1['x'])
    f1_y = np.array(f1['y'])
    x = np.linspace(f1_x.min(), f1_x.max(), 100)
    y = make_interp_spline(f1_x, f1_y)(x)
    plt.plot(x, y, label=label, c=c)
    return plt


if __name__ == '__main__':
    loss = [19, 18, 18, 18, 17, 16]
    label = [i for i in range(1, len(loss) + 1)]

    f1_scores = {'x': [0.1, 0.3, 0.5, 0.8, 1], 'y': [0, 0.1, 0.5, 0.4, 0.02]}
    plt.figure()

    # plt = plot_loss_lr(loss, label, plt, '', '#3bc9db')
    plt = plot_f1_scores(f1_scores, plt, '', '#3bc9db')

    plt.show()
