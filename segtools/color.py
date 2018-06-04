import networkx as nx
import numpy as np

import colorsys
import skimage.io as io
from numba import jit
from skimage import measure
import os
import matplotlib.pyplot as plt

from . import label_tools

## colormaps

def pastel_colors_RGB(n_colors=10, brightness=0.5, value=0.5):
  """
  a cyclic map of equal brightness and value. Good for elements of an unordered set.
  """
  HSV_tuples = [(x * 1.0 / n_colors, brightness, value) for x in range(n_colors)]
  RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
  return RGB_tuples

def pastel_colors_RGB_gap(n_colors=10, brightness=0.5, value=0.5):
  """
  leaves a gap in Hue, so colors don't cycle around, but go from Red to Blue
  """
  HSV_tuples = [(x * 0.75 / n_colors, brightness, value) for x in range(n_colors)]
  RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
  return RGB_tuples

def label_colors(bg_ID=1, membrane_ID=0, n_colors = 10, maxlabel=1000):
  RGB_tuples = pastel_colors_RGB(n_colors=10)
  # intens *= 2**16/intens.max()
  assert membrane_ID != bg_ID
  RGB_tuples *= maxlabel
  RGB_tuples[membrane_ID] = (0, 0, 0)
  RGB_tuples[bg_ID] = (0.01, 0.01, 0.01)
  return RGB_tuples

def rand_cmap_uwe(n=256):
  # cols = np.random.rand(n,3)
  # cols = np.random.uniform(0.1,1.0,(n,3))
  h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
  cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
  cols[0] = 0
  return cols #matplotlib.colors.ListedColormap(cols)

def mpl_color(img, cmap='viridis', mn=None, mx=None):
  if mn is None : mn = img.min()
  if mx is None : mx = img.max()
  cmap = plt.get_cmap(cmap)
  cmap = np.array(cmap.colors)
  # rgb_img = img.copy()
  img = img.copy()
  img = (img-mn)/(mx-mn)
  img = img.clip(min=0,max=1)
  img = ((cmap.shape[0]-1)*img).astype(np.uint8)
  rgb_img = cmap[img.flat].reshape(img.shape + (3,))
  return rgb_img

def grouped_colormap(basecolors=[(1,0,0), (0,1,0)], mult=[100,100]):
  flatten = lambda l: [item for sublist in l for item in sublist]
  colors = flatten([[c] * m for c,m in zip(basecolors, mult)])
  colors = np.array(colors)
  rands = np.random.rand(sum(mult),3)
  colors = colors + rands*0.5
  colors = np.clip(colors, 0, 1)
  return colors

## recoloring / relabeling / mapping labels to new values

def recolor_from_mapping(lab, mapping):
  """
  mapping can be a dictionary of int->value
  value can be int,uint or float type, can be scalar or vector
  """
  maxlabel = lab.max().astype('int')
  somevalue = list(mapping.values())[0]
  if hasattr(somevalue, '__len__'):
    n_channels = len(somevalue)
  else:
    n_channels = 1
  maparray = np.zeros((maxlabel+1, n_channels))
  for k,v in mapping.items():
    maparray[k] = v
  lab2 = maparray[lab.flat].reshape(lab.shape + (n_channels,))
  if lab2.shape[-1]==1: lab2 = lab2[...,0]
  return lab2

def recolor_from_ndarray(lab, perm):
  "perm is an ndarray of length > lab.max()"
  assert perm.shape[0] > lab.max()
  if perm.ndim == 1:
    lab2 = perm[lab.flat].reshape(lab.shape)
  elif perm.ndim == 2:
    lab2 = perm[lab.flat].reshape(lab.shape + (perm.shape[1],))
  else:
    raise TypeError("perm must be 1D or 2D array")
  return lab2

def graphcolor(lab):
  """
  recolor lab s.t. touching labels have (very) different colors.
  you can also assign random color to every object, but graph coloring enforces strong differences.
  """
  matrix = label_tools.pixelgraph_edge_distribution(lab)
  mask = matrix > 0
  pairs = np.indices(matrix.shape)[:,mask].reshape((2,-1)).T
  # g = nx.Graph([(x,y,{'border':res[(x,y)]}) for (x,y) in res.keys()])
  g = nx.Graph([(x,y) for (x,y) in pairs])
  d = nx.coloring.greedy_color(g)
  labr = recolor_from_mapping(lab, d)
  return labr

def mod_color_hypimg(hyp):
  """
  super simple relabeling of hyp; prefer `graphcolor` to this when time permits.
  return an image with values meant for spimagine display
  when plotting labels remember to keep a mask of the zero-values before you mod.
  Then reset zeros to zero after adding 2x the mod value.
  """
  hyp2 = hyp.copy()
  mask = hyp2==0
  hyp2 %= 7
  hyp2 += 5
  hyp2[mask] = 0
  return hyp2

## a simple way of saving / compressing a stack.

def make_jpegfolder_from_stack(rgb, name='rgb'):
  "save a "
  if rgb.ndim==4:
    pass
  elif rgb.ndim==3:
    rgb = cmap_color(rgb)
  if not os.path.exists(name):
    os.makedirs(name)
  for i in range(rgb.shape[0]):
    io.imsave(name + '/' + 'rgb{:03d}.jpeg'.format(i), rgb[i])
