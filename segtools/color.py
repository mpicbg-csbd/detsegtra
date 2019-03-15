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

def pastel_colors_RGB(n_colors=10, max_saturation=1.0, brightness=0.5, value=0.5, bg_id=None, shuffle=True):
  """
  a cyclic map of equal brightness and value. Good for elements of an unordered set.
  """
  HSV_tuples = [(x * max_saturation / n_colors, brightness, value) for x in range(n_colors)]
  cmap = np.array([colorsys.hsv_to_rgb(*x) for x in HSV_tuples])
  if shuffle: np.random.shuffle(cmap)
  if bg_id is not None: 
    cmap[bg_id] = (0,0,0)
  return cmap

def rand_cmap_uwe(n=256):
  h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
  cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
  cols[0] = 0
  return cols

def mpl_color(img, cmap='viridis', mn=None, mx=None):
  if mn is None : mn = img.min()
  if mx is None : mx = img.max()
  cmap = plt.get_cmap(cmap)
  cmap = np.array(cmap.colors)
  img = img.copy()
  img = (img-mn)/(mx-mn)
  img = img.clip(min=0,max=1)
  img = ((cmap.shape[0]-1)*img).astype(np.uint8)
  rgb_img = cmap[img.flat].reshape(img.shape + (3,))
  return rgb_img

def grouped_colormap(basecolors=[(1,0,0), (0,1,0)], mult=[100,100]):
  flatten = lambda l: [item for sublist in l for item in sublist]
  colors  = flatten([[c] * m for c,m in zip(basecolors, mult)])
  colors  = np.array(colors)
  rands   = np.random.rand(sum(mult),3)
  colors  = colors + rands*0.5
  colors  = np.clip(colors, 0, 1)
  return colors

## recoloring / relabeling / mapping labels to new values

def recolor_from_mapping(lab, mapping):
  """
  mapping can be a dictionary of int->value
  value can be int,uint or float type, can be scalar or vector
  """
  assert set.issubset(set(mapping.keys()), set(np.unique(lab)))
  maxlabel = lab.max().astype('int')
  maparray = np.zeros((maxlabel+1,3))
  for k,v in mapping.items():
    maparray[k] = v
  lab2 = maparray[lab.flat].reshape(lab.shape + (3,))
  return lab2

def relabel_from_mapping(lab, mapping, setzero=False):
  assert set.issubset(set(mapping.keys()), set(np.unique(lab)))
  maxlabel = lab.max().astype('int')
  maparray = np.zeros(maxlabel+1) if setzero else np.arange(maxlabel+1)
  for k,v in mapping.items():
    maparray[k] = v
  lab2 = maparray[lab.flat].reshape(lab.shape)
  return lab2

## too simple. use the one-liner instead.
@DeprecationWarning
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
  recolor lab s.t. touching labels have different colors, even if total number of colors is small.
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
