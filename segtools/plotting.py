import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from . import patchmaker
from . import lib


def nuc_grid_plot(img, nhl):
    if len(nhl) > 100:
        print("Too many nuclei! Try again w len(nhl) < 100.")
        return False

    def f(i):
        img_crop = lib.nuc2img(nhl[i], img, 4)
        lab, ncells = lib.label(img_crop > 0.92)
        lab = lab.sum(2)
        return lab
    patches = [f(i) for i in range(len(nhl))]

    coords = np.indices((4, 5))*30
    plotimg = patchmaker.piece_together_ragged_2d(
        patches, coords.reshape(2, -1).T)
    plt.imshow(plotimg)

def nhl2crops(img, nhl, axis=None, pad=10):
    def f(i):
        ss = lib.nuc2slices(nhl[i], pad, shift=pad)
        img_crop = img[ss].copy()
        if img_crop.ndim==3 and axis is not None: 
            print('hooligans', ss)
            a = img_crop.shape[axis]
            img_crop=img_crop[a//2]
        return img_crop
    patches = [f(i) for i in range(len(nhl))]
    return patches

def plot_nhls(nhls,
                x=lambda n:n['coords'][0], 
                y=lambda n:np.log2(n['area'])):
    cm = sns.cubehelix_palette(len(nhls))
    for i,nhl in enumerate(nhls):
        xs = [x(n) for n in nhl]
        ys = [y(n) for n in nhl]
        plt.scatter(xs, ys, c=cm[i])

def ax_scatter_data(ax, data, **kwargs):
    """
    matplotlib scatterplot, but you can use a list of dicts
    most common labels are plotted first, (underneath).
    """
    xs = np.array([d['x'] for d in data])
    ys = np.array([d['y'] for d in data])
    cs = np.array([d['c'] for d in data])
    szs = np.array([d['s'] for d in data])
    labels = np.array([d['l'] for d in data])

    print(kwargs)

    # sort labels from most to least frequent
    labelset, labelcts = np.unique(labels, return_counts=True)
    inds = np.argsort(labelcts)

    for l in labelset[inds][::-1]:
        mask = labels == l
        ax.scatter(xs[mask], ys[mask], s=szs[mask],
                   c=cs[mask], label=l, **kwargs)

def ax_scatter_plus(ax, xs, ys, cs, labels, szs):
    """
    matplotlib scatterplot, but you can use a list of labels
    most common labels are plotted first, (underneath).
    """
    xs = np.array(xs)
    ys = np.array(ys)
    cs = np.array(cs)
    szs = np.array(szs)

    # sort labels from most to least frequent
    labels = np.array(labels)
    labelset, labelcts = np.unique(labels, return_counts=True)
    inds = np.argsort(labelcts)

    for l in labelset[inds][::-1]:
        mask = labels == l
        ax.scatter(xs[mask], ys[mask], s=szs[mask], c=cs[mask], label=l)

def lineplot(img):
    pal = sns.diverging_palette(255, 133, l=60, n=7, center="dark")
    sns.set_palette(pal)
    a, b, c = img.shape
    lines = img[a//2, ::100].reshape(-1, c)
    fig = plt.figure()
    ax = fig.gca()
    for l in lines:
        ax.plot(l)
    ax2 = fig.add_axes([0.23, 0.50, 0.3, 0.3])
    ax2.imshow(img.max(0))
