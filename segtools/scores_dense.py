import numpy as np
from numba import jit
from scipy.ndimage import label

@jit
def pixel_sharing_bipartite(lab1, lab2):
  assert lab1.shape == lab2.shape
  psg = np.zeros((lab1.max()+1, lab2.max()+1), dtype=np.int)
  for i in range(lab1.size):
    psg[lab1.flat[i], lab2.flat[i]] += 1
  return psg

## bipartite statistics

def bipartite_entropy(psg, axis=0):
  "entropy for each object in axis. useful for recoloring."
  axis = 1-axis
  entropy = psg/psg.sum(axis=axis, keepdims=True)
  entropy = np.where(entropy!=0, entropy * np.log2(entropy), 0)
  entropy = -entropy.sum(axis=axis)
  return entropy

## weighted bipartite graphs in matrix form

def intersection_over_union(psg):
  rsum = np.sum(psg, 0, keepdims=True)
  csum = np.sum(psg, 1, keepdims=True)
  return psg / (rsum + csum - psg)

## matchings from psg

def matching_overlap(psg, fractions=(0.5,0.5)):
  """
  create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
  NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
  NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
  """
  afrac, bfrac = fractions
  set0_object_sizes = np.sum(psg, axis=1, keepdims=True)
  m0  = np.where(set0_object_sizes==0,0,psg / set0_object_sizes)
  set1_object_sizes = np.sum(psg, axis=0, keepdims=True)
  m1 = np.where(set1_object_sizes==0,0,psg / set1_object_sizes)
  m0 = m0 > afrac
  m1 = m1 > bfrac
  matching = m0 * m1
  matching = matching.astype('bool')
  return matching

def matching_iou(psg, fraction=0.5):
  iou = intersection_over_union(psg)
  matching = iou > 0.5
  matching[:,0] = False
  matching[0,:] = False
  return matching

def matching_max(psg):
  """
  matching based on mutual first preference
  """
  rowmax = np.argmax(psg, axis=0)
  colmax = np.argmax(psg, axis=1)
  starting_index = np.arange(psg.shape[1])
  equal_matches = colmax[rowmax[starting_index]]==starting_index
  rm, cm = rowmax[equal_matches], colmax[rowmax[equal_matches]]
  matching = np.zeros_like(psg)
  matching[rm, cm] = 1
  return matching

## full scores

def seg(lab_gt, lab, partial_dataset=False):
  """
  calculate seg from pixel_sharing_bipartite
  seg is the average conditional-iou across ground truth cells
  conditional-iou gives zero if not in matching
  ----
  calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
  for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
  IoU as low as 1/2 that don't match, and thus have CIoU = 0.
  """
  psg = pixel_sharing_bipartite(lab_gt, lab)
  iou = intersection_over_union(psg)
  matching = matching_overlap(psg, fractions=(0.5, 0))
  matching[0,:] = False
  matching[:,0] = False
  n_gt = len(set(np.unique(lab_gt)) - {0})
  n_matched = iou[matching].sum()
  if partial_dataset:
    return n_matched , n_gt
  else:
    return n_matched / n_gt

def precision(lab_gt, lab, iou=0.5, partial_dataset=False):
  """
  precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
  """
  psg = pixel_sharing_bipartite(lab_gt, lab)
  matching = matching_iou(psg, fraction=iou)
  assert matching.sum(0).max() < 2
  assert matching.sum(1).max() < 2
  n_gt  = len(set(np.unique(lab_gt)) - {0})
  n_hyp = len(set(np.unique(lab)) - {0})
  n_matched = matching.sum()
  if partial_dataset:
    return n_matched , (n_gt + n_hyp - n_matched)
  else:
    return n_matched / (n_gt + n_hyp - n_matched)

## objects for matchings ... sets, maps and masks

@DeprecationWarning
def matching_sets(lab_gt, lab):
  m1,m2 = matching_maps(lab_gt, lab)
  s1 = set(m1.keys())
  s2 = set(m2.keys())
  s1c = (set(np.unique(lab_gt)) - {0}) - s1
  s2c = (set(np.unique(lab))    - {0}) - s2
  return s1,s2,s1c,s2c

@DeprecationWarning
def matching_maps(lab_gt, lab):
  psg = pixel_sharing_bipartite(lab_gt, lab)
  matching = matching_iou(psg, fraction=0.5)
  assert matching.sum(0).max()<2
  assert matching.sum(1).max()<2
  map1, map2 = maps_from_matching(matching)
  return map1, map2

@DeprecationWarning
def matching_masks(lab_gt, lab):
  """
  mask1 = objects in gt that match to lab
  mask2 = objects in lab that match to gt
  mask1c = objects in gt that DONT match to lab
  mask1c = objects in lab that DONT match to gt
  """
  s1,s2,s1c,s2c = matching_sets(lab_gt, lab)
  # from . import nhl_tools as label_tools
  mask1  = mask_labels(s1, lab_gt)
  mask2  = mask_labels(s2, lab)
  mask1c = mask_labels(s1c, lab_gt)
  mask2c = mask_labels(s2c, lab)
  return mask1, mask2, mask1c, mask2c

def mask_labels(labels, lab):
  mask = lab.copy()
  recolor = np.zeros(lab.max()+1, dtype=np.bool)
  for l in labels:
    recolor[l] = True
  mask = recolor[lab.flat].reshape(lab.shape)
  return mask

def sets_maps_masks_from_matching(lab_gt, lab, matching):
  """
  assumes bg == 0
  """
  map1, map2 = maps_from_matching(matching)
  s1 = set(map1.keys())
  s2 = set(map2.keys())
  s1c = (set(np.unique(lab_gt)) - {0}) - s1
  s2c = (set(np.unique(lab))    - {0}) - s2
  mask1  = mask_labels(s1, lab_gt)
  mask2  = mask_labels(s2, lab)
  mask1c = mask_labels(s1c, lab_gt)
  mask2c = mask_labels(s2c, lab)
  res = {}
  res['maps'] = (map1, map2)
  res['sets'] = (s1,s2,s1c,s2c)
  res['masks'] = (mask1, mask2, mask1c, mask2c)
  return res

def maps_from_matching(matching):
  """
  matching between all nonzero ids
  assumes bg == 0
  """
  assert matchingQ(matching)
  map1 = dict()
  map2 = dict()
  for i in range(1, matching.shape[0]):
    m = matching[i].argmax()
    if m > 0:
      map1[i] = m
      map2[m] = i
  return map1, map2

## fix label images

def find_split_labels(img):
  """
  assumes default definition of connected components. in 2D this is 4-connection. in 3D this is 6-connection.
  assumes 0-valued background (bg may be unconnected)
  returns: set of split label ids
  """
  s = set()
  for l in set(np.unique(img))-{0}:
    lab = label(img==l)[0]
    ncs = len(np.unique(lab))
    if ncs>2:
      s.add(l)
  return s

def fix_split_labels(img, labelset):
  for l in labelset:
    lab = label(img==l)[0]
    for j in range(2,lab.max()+1):
      img[lab==j] = img.max()+1

def make_dense(lab):
  "might have to run with denseQ multiple times!"
  for l in denseQ(lab):
    lab[lab==lab.max()] = l

def identify_background(psg):
  "just in case it's not zero-labeled. only works with dense labeling."
  memlabel = np.argmax((psg!=0).sum(0))
  return memlabel

## predicates

def denseQ(lab):
  "important: doesn't return true/false, but works as predicate as set() evals to False."
  return set(np.arange(lab.min(), lab.max()+1)) - set(np.unique(lab))

def matchingQ(matching):
  b0 = matching.dtype in [np.bool, np.uint8, np.uint16, np.uint32, np.uint64]
  b1 = np.sum(matching,0).max() == np.sum(matching,1).max() <= 1
  if not b0:
    raise TypeError("Matching should be bool or uint type.")
  return b1
