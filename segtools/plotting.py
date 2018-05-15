import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from . import patchmaker
from . import lib
from . import segtools_simple as ss
from . import color
from . import voronoi


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


def make_comparison_image(img_raw, lab, lab_gt, ax=None):
  # cmap_gt  = np.array([(0,0,0)] + sns.color_palette('hls', lab_gt.max()))
  # cmap_gt = np.zeros((3000,4), np.float)
  # cmap_gt[:,[0,1,2]] = rand_cmap(3000)
  # cmap_gt[1:,3] = 1
  # lab_gt_recolor = cmap_gt[lab_gt.flat].reshape(lab_gt.shape + (4,))

  if ax is None:
    ax = plt.gca()

  colormap = plt.cm.Greys_r
  vm = np.percentile(img_raw, 99.5)

  # compute masks

  psg = ss.pixel_sharing_bipartite(lab_gt, lab)
  matching = ss.matching_overlap(psg, fractions=(0.5, 0.5))
  matchstuff = ss.sets_maps_masks_from_matching(lab_gt, lab, matching)
  m1, m2, m1c, m2c = matchstuff['masks']
  map1, map2 = matchstuff['maps']
  labcopy = color.apply_mapping(lab, map2)
  if m2c.sum() > 0:
    lab[m2c] = lab[m2c] - lab[m2c].min() + lab_gt[m1].max() + 1
  lab[m2] = labcopy[m2]
  
  # relabel and color

  grb = lab.copy()
  ggt = lab_gt.copy()
  borders_lab = voronoi.lab2binary_neibs(grb)
  borders_lab = borders_lab != 4
  borders_gt = voronoi.lab2binary_neibs(ggt)
  borders_gt = borders_gt != 4

  grb[~borders_lab] = 0
  grb[borders_lab & m2] = 1
  grb[borders_lab & m2c] = 2
  green = (0,1,0,.7)
  lightgreen = (0,1,0,.25)
  red   = (1,0,0,.7)
  trans = (0,0,0,0.0)
  lightblue  = (0,0,1,0.25)

  cmap = np.array([trans, green, red], dtype=np.float)

  grb  = cmap[grb.flat].reshape(grb.shape + (4,))
  
  res = ax.imshow(img_raw, cmap=colormap, vmax=vm)
  ax.imshow(grb)

  # Add Overlay

  # if False:
  grb = lab_gt.copy()
  grb[~m1c] = 0
  grb[m2]   = 2
  grb[m1c]  = 3

  cmap = np.array([trans, trans, lightgreen, lightblue])
  grb = cmap[grb.flat].reshape(grb.shape + (4,))
  ax.imshow(grb)
  return ax
