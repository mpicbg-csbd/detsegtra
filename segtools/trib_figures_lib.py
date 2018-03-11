"""
ALL DEPRECATED!!!
"""

import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import numpy as np
import skimage.io as io
from scipy.ndimage import label

import gputools

from . import lib
from . import segtools

def jaccard_color(img, mn=0.5):
  cmap = plt.get_cmap('viridis')
  cmap = np.array(cmap.colors)
  rgb_img = img.copy()
  mx = rgb_img.max()
  rgb_img -= mn
  rgb_img *= mx/rgb_img.max()
  rgb_img = rgb_img.clip(min=0)
  # mn, mx = rgb_img.min(), rgb_img.max()
  # rgb_img = (rgb_img - mn)/(mx-mn)
  rgb_img = (255*rgb_img).astype(np.uint8)
  rgb_img = cmap[rgb_img.flat].reshape(rgb_img.shape + (3,))
  return rgb_img

def rmsmall(hyp, minsize=27):
  "inplace!"
  nhl = lib.hyp2nhl(hyp)
  small = [n for n in nhl if n['area'] < minsize]
  mask = lib.mask_nhl(small, hyp)
  hyp[mask] = 0

# def ratio2(hyp, hyp_gt):
#   "compure the matching score for a single hypothesis image. minsize filters out small nuclei."
#   res = segtools.stats_seg_matching(hyp_gt, hyp)
#   return res['stats']['ratio_2']

def colorit(hyp, res, hyp_gt):
  hyp = hyp.copy()
  mask_matched = (0 < hyp) & (hyp <= hyp_gt.max())
  unmatched_gt_set = set(hyp_gt.flat) - res['matched_gt_set'] - {0}
  mask_unmatched_gt = lib.mask_labels(unmatched_gt_set, hyp_gt)
  mask_unmatched_seg = hyp_gt.max() < hyp
  hyp[mask_matched] = 1
  hyp[mask_unmatched_gt] = 2
  hyp[mask_unmatched_seg] = 3
  return hyp

def colorit_multicolorRGB(hyp, res, hyp_gt, with_unmatched_gt=False, axis=0):
  hyp = hyp.copy()
  mask_matched = (0 < hyp) & (hyp <= hyp_gt.max())
  unmatched_gt_set = set(hyp_gt.flat) - res['matched_gt_set'] - {0}
  mask_unmatched_gt = lib.mask_labels(unmatched_gt_set, hyp_gt)
  mask_unmatched_seg = hyp_gt.max() < hyp
  
  if with_unmatched_gt:
    hyp[mask_matched] = (hyp[mask_matched] % 7) + 1 # in 1..7
    hyp[mask_unmatched_gt] = 8
    hyp[mask_unmatched_seg] = 9 # note that the numerical order determines which appears on top in the max projections!
    hyp = hyp.max(axis)
    cmap = [(1,1,1)] + sns.color_palette('hls', 7) + [(1,0,0), (0,0,0)]
  else:
    hyp[mask_matched] = (hyp[mask_matched] % 7) + 1 # in 1..7
    hyp[mask_unmatched_seg] = 8
    hyp = hyp.max(axis)
    cmap = [(1,1,1)] + sns.color_palette('hls', 7) + [(0,0,0)]

  cmap = np.array(cmap)
  hyp = cmap[hyp.flat].reshape(hyp.shape + (3,))
  return hyp

def colorit_greyredblackRGB(hyp, res, hyp_gt, axis=0):
  hyp = hyp.copy()
  mask_matched = (0 < hyp) & (hyp <= hyp_gt.max())
  unmatched_gt_set = set(hyp_gt.flat) - res['matched_gt_set'] - {0}
  mask_unmatched_gt = lib.mask_labels(unmatched_gt_set, hyp_gt)
  mask_unmatched_seg = hyp_gt.max() < hyp
  
  hyp[mask_matched] = 1
  hyp[mask_unmatched_gt] = 2
  hyp[mask_unmatched_seg] = 3
  
  hyp = hyp.max(axis)
  cmap = [(1,1,1), (0.7, 0.7, 0.7), (1,0,0), (0,0,0)]
  cmap = np.array(cmap)
  hyp = cmap[hyp.flat].reshape(hyp.shape + (3,))
  return hyp