# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    # if smooth == 1:
    #     x, y = smooth_reward_curve(x, y)
    #
    # if smooth == 2:
    #     y = medfilt(y, kernel_size=9)
    #
    # x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def visdom_plot(viz, win, value, reward, state, folder, game, name, num_steps, bin_size=10, smooth=1):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        num_steps = len(y)
        y = np.concatenate((np.ones(box_pts - 1) * y[0], y), axis=0)
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth

    def STD_fn(y, box_pts):
        num_steps = len(y)
        y = np.concatenate((np.ones(box_pts - 1) * y[0], y), axis=0)


        return np.std([y[ii: ii + box_pts]
                      for ii in range(num_steps)], axis=1)

    # Plot reward and mean ######################
    fig = plt.figure()
    plt.tight_layout()
    smooth_reward = smooth(ty, 10)

    plt.plot(tx, ty, label="Reward", color='r')
    plt.plot(tx, smooth_reward, label="Mean Reward", color='b')
    # plt.fill_between(tx, smooth_reward - std_reward,
    #                  smooth_reward + std_reward, color='b', alpha=0.2)

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    # plt.xticks(ticks, tick_names)
    plt.xlim(0, tx[-1] * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # plot mean reward and std ##########################
    fig = plt.figure()

    plt.tight_layout()
    std_reward = STD_fn(ty, 10)

    plt.plot(tx, smooth_reward, label="Average Reward", color='r')
    plt.fill_between(tx, smooth_reward - std_reward,
                     smooth_reward + std_reward, color='b', alpha=0.2)

    plt.xlim(0, tx[-1] * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image_std = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image_std = image_std.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # plot Value and State for value
    fig = plt.figure()

    ax = plt.subplot(1, 3, 1)
    im = ax.imshow(value[0,], cmap='viridis')
    plt.title('Value')
    plt.colorbar(im)
    ax = plt.subplot(1, 3, 2)
    im = ax.imshow(reward[0,], cmap='viridis')
    plt.title('Reward')
    plt.colorbar(im)
    ax = plt.subplot(1, 3, 3)
    ax.imshow((state * 255).permute(1,2,0))
    plt.title('State')
    plt.show()
    plt.draw()

    image_value_state = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image_value_state = image_value_state.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)



    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    image_std = np.transpose(image_std, (2, 0, 1))
    image_value_state = np.transpose(image_value_state, (2, 0, 1))
    win[0] = viz.image(image, win=win[0])
    win[1] = viz.image(image_std, win=win[1])
    win[2] = viz.image(image_value_state, win=win[2])
    return win


if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom()
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
