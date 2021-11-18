import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import os


def plot_trajectory(ts, ax, plot_3d=False, color_code=None, alpha=1.):
    if color_code is not None:
        colors = [color_code] * len(ts)
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
    for i in range(len(ts) - 1):
        if plot_3d:
            ax.plot(ts[i:i+2, 0], ts[i:i+2, 1], ts[i:i+2, 2],
                    marker='o', c=colors[i], alpha=alpha)
        else:
            ax.plot(ts[i:i+2, 0], ts[i:i+2, 1],
                    marker='o', c=colors[i], alpha=alpha)


def set_fig_style(font_size=22):
    os.environ['PATH'] += ':/Library/TeX/texbin/'  # Path to your latex install
    rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
    rc('font', family="serif", serif="Times", size=font_size)
    rc('text', usetex=True)


def get_rot2d(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )


def get_rot3d(alpha, beta, gamma):
    R_alpha = np.array(
        [[1., 0., 0.],
         [0., np.cos(alpha), -np.sin(alpha)],
         [0., np.sin(alpha), np.cos(alpha)]]
    )
    R_beta = np.array(
        [[np.cos(beta), 0., np.sin(beta)],
         [0., 1., 0.],
         [-np.sin(beta), 0., np.cos(beta)]]
    )
    R_gamma = np.array(
        [[np.cos(gamma), -np.sin(gamma), 0.],
         [np.sin(gamma), np.cos(gamma), 0.],
         [0., 0., 1.]]
    )
    return R_alpha.dot(R_beta).dot(R_gamma)


def make_one_spiral(sz, noise=.5):
    uniform_in_01 = np.random.rand(sz, 1)
    non_uniform_in_01 = np.power(uniform_in_01, 4)
    n = np.sqrt(non_uniform_in_01) * 780 * (2 * np.pi) / 360
    n = np.sort(n.reshape((-1, ))).reshape((-1, 1))
    d1x = -np.cos(n) * n + np.random.rand(sz, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(sz, 1) * noise
    arr = np.array(np.hstack((d1x, d1y)))
    return arr / np.max(n)


def make_spirals(n, sz, noise=.5, shift=False, some_3d=False):
    dataset = []
    for i in range(n):
        if some_3d and i % 2 == 0:
            spiral = make_one_spiral(sz=sz, noise=0.)
            spiral = np.hstack((spiral, noise * np.random.randn(sz, 1)))
            alpha = (np.random.rand(1)[0] - .5) * np.pi / 2
            beta = (np.random.rand(1)[0] - .5) * np.pi / 2
            gamma = (np.random.rand(1)[0] - .5) * np.pi / 2
            spiral = np.dot(spiral, get_rot3d(alpha, beta, gamma))
            if shift:
                spiral += np.random.rand(3) * 3
        else:
            spiral = make_one_spiral(sz=sz, noise=noise)
            theta = np.random.rand(1)[0] * 2 * np.pi
            spiral = np.dot(spiral, get_rot2d(theta))
            if shift:
                spiral += np.random.rand(2) * 3
        dataset.append(spiral)
    return dataset


def make_one_folium(sz, a=1., noise=.1, resample_fun=None):
    theta = np.linspace(0, 1, sz)
    if resample_fun is not None:
        theta = resample_fun(theta)
    theta -= .5
    theta *= .9 * np.pi
    theta = theta.reshape((-1, 1))
    r = a / 2 * (4 * np.cos(theta) - 1. / np.cos(theta))
    x = r * np.cos(theta) + np.random.rand(sz, 1) * noise
    y = r * np.sin(theta) + np.random.rand(sz, 1) * noise
    return np.array(np.hstack((x, y)))


def make_folia(n, sz, noise=.1, shift=False, some_3d=False):
    if some_3d:
        raise NotImplementedError
    dataset = []
    for _ in range(n):
        spiral = make_one_folium(sz=sz, a=1., noise=noise)
        theta = np.random.rand(1)[0] * 2 * np.pi
        spiral = np.dot(spiral, get_rot2d(theta))
        if shift:
            spiral += np.random.rand(2) * 3
        dataset.append(spiral)
    return dataset
