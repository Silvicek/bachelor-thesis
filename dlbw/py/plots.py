# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
mpl.use('pgf')
def figsize(scale):
    fig_width_pt = 363.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    # golden_mean = (np.sqrt(5.0)-1.0)/3
    # golden_mean = .6  # ou
    golden_mean = .4  # activations
    # golden_mean = .4  # compare runs
    # golden_mean = 0.7  # joints

    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "lmss",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "text.fontsize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


def strip_zeros(array):
    for i in range(1, len(array)):
        if np.count_nonzero(array[i, :]) == 0:
            return array[:i, :]
    return array


def plot(y, p):
    f, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, sharey=True)
    ax0.plot(y[:, 0])
    ax1.plot(y[:, 2])
    ax2.plot(y[:, 4])
    ax3.plot(y[:, 6])
    ax4.plot(y[:, 8])
    ax5.plot(y[:, 10])
    ax0.plot(p[:, 0])
    ax1.plot(p[:, 2])
    ax2.plot(p[:, 4])
    ax3.plot(p[:, 6])
    ax4.plot(p[:, 8])
    ax5.plot(p[:, 10])
    plt.show()


def plotone(y, i=1, show=False):
    plt.figure(i)
    y = strip_zeros(y)
    for i, v in enumerate(xrange(6)):
        v += 1
        ax1 = plt.subplot(6, 1, v)
        x = np.linspace(0, 0.05 * len(y[:, i]), len(y[:, i]))
        ax1.plot(x, y[:, i * 2], )
    if show:
        plt.show()


def plot_t_r(path):
    x = np.load(open(path))
    plt.plot(x[0], x[1])
    plt.show()


from ddpg.explorers import OrnsteinUhlenbeck

def myround(x, base, up=False):
    if up:
        return base * round(float(x)/base)
    return base * round(float(x)/base)

plots_path = '/Desktop/neural-networks-for-humanoid-robot-control/plots/'
def plots(shape, data, axis=None, path=None, labels=None, legends=None):
    # """data=[(x1,y1),...]"""
    """data=[([x10,x11,..],[y10,y11,..]),...]
    labels=string or [(x1,y1), (x2,y2),...]
    legends=
    """
    fig, axes = plt.subplots(*shape, sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    try:
        axes = [item for sublist in axes for item in sublist]
    except TypeError:
        pass
    p_ = fig.add_subplot(111, frameon=False)
    p_.axes.get_xaxis().set_ticks([])
    p_.axes.get_yaxis().set_ticks([])

    for ax, (x, y), i in zip(axes, data, range(len(axes))):
        for x_, y_ in zip(x, y):
            ax.plot(x_, y_, linewidth=.5)
        if axis:
            ax.axis(axis)
        if isinstance(labels, list):
            if labels[i][0] != '':
                ax.set_xlabel(labels[i][0])
            if labels[i][1] != '':
                ax.set_ylabel(labels[i][1])
        if isinstance(legends, list):
            ax.legend(legends[i], loc=2)
            # ax.legend(legends[i], loc=2, ncol=3)

        ax.grid(True)
        # ax.set_yticklabels([])


    if not isinstance(labels, list) and labels is not None:
        xlabel = labels.split('___')[0]
        ylabel = labels.split('___')[-1]
        p_.set_xlabel(xlabel, labelpad=20)
        if xlabel != ylabel:
            p_.set_ylabel(ylabel, labelpad=30)
    fig.tight_layout()
    if path:
        from os.path import expanduser
        home = expanduser("~")
        fig.savefig(home+plots_path+path+'.pgf')
        fig.savefig(home+plots_path+path+'.pdf')

    plt.show()


def ou(name):
    dt = 1e-2
    T = 20.
    ou = OrnsteinUhlenbeck(1, dt, T, theta=.15, sigma=.2)
    t = np.linspace(0., T, T/dt)
    data = []
    for i in range(4):
        x = ou.noise.flatten()
        data.append(([t], [x]))
        ou.reset()

    mi = myround(min([min(y[0]) for x,y in data]),.5)
    ma = myround(max([max(y[0]) for x,y in data]),.5, True)

    axis = [0,20,-1,1]
    path = name
    labels = 't[s]'
    plots((2,2),data, axis=axis, path=path, labels=labels)

def activations(path):
    t = np.linspace(-3,3,100)
    sigm = 1./(1+np.exp(-t))
    tanh = np.tanh(t)
    relu = np.maximum(t, 0)
    data = [([t], [sigm]), ([t], [tanh]), ([t], [relu])]
    plots((1,3), data, path=path, axis=[-3,3,-1.2,1.2])

def noise(path):
    no_noise = np.load(open('pendulum/swingup/10x10_100x100_no_noise/plots/plot.npy'))
    std_noise = np.load(open('pendulum/swingup/10x10_100x100_std/plots/plot.npy'))
    x3_noise = np.load(open('pendulum/swingup/10x10_100x100_3x_noise/plots/plot.npy'))
    no_noise[0] /= 1e3
    std_noise[0] /= 1e3
    x3_noise[0] /= 1e3

    data = [([no_noise[0], std_noise[0], x3_noise[0]], [no_noise[1], std_noise[1], x3_noise[1]])]
    legends = [(['zero noise','noise','noise x3'])]
    plots((1,1), data, path=path, axis=[0,2500,-20,60], legends=legends)


def adam_sgd(path):
    adam = np.load(open('brian/swingup/50x50-100x100_last_false/plots/plot.npy'))
    sgd = np.load(open('brian/swingup/50x50-100x100_sgd/plots/plot.npy'))
    adam[0] = adam[0]/1e3-30
    sgd[0] = sgd[0]/1e3-30

    data = [([adam[0], sgd[0]], [adam[1], sgd[1]])]
    legends = [(['Adam','SGD'])]
    labels = [('\# 1000s learning steps','score')]
    plots((1,1), data, path=path, axis=[0,2500,-10,60], legends=legends,
          labels=labels)

def noise_test(path):
    adam = np.load(open('brian/swingup/50x50-100x100_last_false/plots/plot.npy'))
    sgd = np.load(open('brian/swingup/50x50-100x100_no_noise/plots/plot.npy'))
    adam[0] = adam[0]/1e3-30
    sgd[0] = sgd[0]/1e3-30

    data = [([adam[0], sgd[0]], [adam[1], sgd[1]])]
    legends = [(['Standard run','No noise'])]
    labels = [('\# 1000s learning steps','score')]
    plots((1,1), data, path=path, axis=[0,2500,-10,60], legends=legends,
          labels=labels)

def last_false(path):
    adam = np.load(open('brian/swingup/50x50-100x100_std/plots/plot.npy'))
    sgd = np.load(open('brian/swingup/50x50-100x100_last_false/plots/plot.npy'))
    adam[0] = adam[0]/1e3-30
    sgd[0] = sgd[0]/1e3-30

    data = [([adam[0], sgd[0]], [adam[1], sgd[1]])]
    legends = [(['std','last_false'])]
    labels = [('','')]
    plots((1,1), data, path=path, axis=[0,2500,-10,60], legends=legends,
          labels=labels)

def act_off(path):
    std = np.load(open('brian/swingup/50x50-100x100_last_false/plots/plot.npy'))
    croff = np.load(open('brian/swingup/50x50-0/plots/plot.npy'))
    actoff = np.load(open('brian/swingup/0-100x100/plots/plot.npy'))
    std[0] = std[0]/1e3-30
    croff[0] = croff[0]/1e3-30
    actoff[0] = actoff[0]/1e3-30

    data = [([std[0], croff[0], actoff[0]], [std[1], croff[1], actoff[1]])]
    legends = [(['Standard run','Small critic', 'Small actor'])]
    labels = [('\# 1000s learning steps','score')]
    plots((1,1), data, path=path, axis=[0,2500,-10,60], legends=legends,
          labels=labels)

def unroll(plot_data, x=None):
    # data=[([x10,x11,..],[y10,y11,..]),...]
    data = []
    print plot_data[0].shape
    for y1,y2 in zip(plot_data[0],plot_data[1]):
        data.append(([np.arange(len(y1))]*2,[y1,y2],))
    return data

def joints(path):
    # data=[([x10,x11,..],[y10,y11,..]),...]
    x = np.load('data/playbackx_pos.npy')
    y = np.load('data/playbacky_pos.npy')
    y = y[:, 1:251]
    a_orig = np.array(y) / 2.

    y2 = np.load('data/rnn_10_learnedy.npy')[:250].T*2
    # y2 = np.transpose(y[:, 1:])[0:250]
    leg1 = np.array([data for i,data in enumerate(y2) if i % 2 == 0][1:])
    leg2 = np.array([data for i,data in enumerate(y2) if i % 2 == 1][1:])
    data = unroll([leg1, leg2])
    plots((5,1), data, path=path)  # , labels='\\# iterations___angle[rad]'


def prior_comp(path):
    adam = np.load(open('brian/swingup/50x50-100x100_last_false/plots/plot.npy'))
    sgd = np.load(open('brian/swingup/prior_final_23_5/plots/plot.npy'))
    adam[0] = adam[0]/1e3-30
    sgd[0] = sgd[0]/1e3-30

    data = [([adam[0], sgd[0]], [adam[1], sgd[1]])]
    legends = [(['Standard run','Prioritized replay'])]
    labels = [('\# 1000s learning steps','score')]
    plots((1,1), data, path=path, axis=[0,2500,-10,60], legends=legends,
          labels=labels)


if __name__ == '__main__':
    # joints('supervised/joints2')
    # ou('ddpg/ou')
    # adam_sgd('ddpg/adam_sgd')
    # noise_test('ddpg/noise_comp')
    # act_off('ddpg/act_off')
    # prior_comp('ddpg/prior')
    # plot_t_r('../models/plot.npy')
    # activations('supervised/activations')
    # noise('supervised/test')
    print 'exit'


    # plot_t_r('')
    # plot_t_r('plots/pendulum_balance.npy')
    # plot_t_r('plots/pendulum_swingup.npy')
    # plot_t_r('plots/pendulum_swingup_replayprior400x300.npy')

