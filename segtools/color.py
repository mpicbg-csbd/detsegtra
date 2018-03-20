import colorsys
import networkx as nx
import numpy as np
import skimage.io as io
import colorsys
from numba import jit
from skimage import measure

from . import voronoi

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

def labelImg_to_rgb(img, bg_ID=1, membrane_ID=0):
  """
  TODO: merge this with the numba version from cell_tracker
  """
  # TODO: the RGB_tuples list we generate is 10 times longer than it needs to be
  RGB_tuples = label_colors(bg_ID, membrane_ID, n_colors=10, maxlabel=img.max())
  a,b = img.shape
  rgb = np.zeros((a,b,3), dtype=np.float32)
  for val in np.unique(img):
      mask = img==val
      print(mask.shape)
      # rgb[mask,:] = np.array(get_color_from_label(val))
      rgb[mask,:] = RGB_tuples[val]
  # f16max = np.finfo(np.float16).
  print(rgb.max())
  # rgb *= 255*255
  return rgb.astype(np.float32) # Preview on Mac only works with 32bit or lower :)

def apply_mapping(lab, mapping):
  maxlabel = lab.max().astype('int')
  maparray = np.zeros(maxlabel+1, dtype=np.uint32)
  for k,v in mapping.items():
    maparray[k] = v
  lab2 = maparray[lab.flat].reshape(lab.shape)
  return lab2

def permute(lab, perm):
  if perm.ndim == 1:
    lab2 = perm[lab.flat].reshape(lab.shape)
  elif perm.ndim == 2:
    lab2 = perm[lab.flat].reshape(lab.shape + (perm.shape[1],))
  else:
    raise TypeError("perm must be 1D or 2D array")
  return lab2

def graphcolor(lab):
  res = voronoi.label_neighbors(lab, ndim=2)
  g = nx.Graph([(x,y,{'border':res[(x,y)]}) for (x,y) in res.keys()])
  d = nx.coloring.greedy_color(g)
  labr = apply_mapping(lab, d)
  return labr


def mod_color_hypimg(hyp):
  """
  give a hypothesis image
  return an image with values meant for spimagine display
  when plotting labels. Remember to keep a mask of the zero-values before you mod. Then reset zeros to zero after adding 2x the mod value.
  """
  hyp2 = hyp.copy()
  mask = hyp2==0
  hyp2 %= 7
  hyp2 += 5
  hyp2[mask] = 0
  return hyp2