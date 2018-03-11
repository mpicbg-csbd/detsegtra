import numpy as np
from numba import jit
from scipy.ndimage import label

def find_split_labels(img):
  """
  assumes default definition of connected components. in 2D this is 4-connection. in 3D this is 6-connection.
  """
  s = set()
  for l in set(np.unique(img))-{0}:
    mask = img==l
    lab = label(mask)[0]
    ncs = len(np.unique(lab))
    if ncs>2:
      s.add(l)
  return s

def fiximg(img, labelset):
  for l in labelset:
    lab = label(img==l)[0]
    for j in range(2,lab.max()+1):
      img[lab==j] = img.max()+1

def seg(lab_gt, lab_seg, partial_dataset=False):
  """
  calculate seg from pixel_sharing_bipartite
  seg is the average conditional-iou across ground truth cells
  conditional-iou gives zero if not in matching
  ----
  calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
  for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
  IoU as low as 1/2 that don't match, and thus have CIoU = 0.
  """
  psg = pixel_sharing_bipartite(lab_gt, lab_seg)
  iou = intersection_over_union(psg)
  matching = matching_overlap(psg, fractions=(0.5, 0))
  matching[0,:] = False
  matching[:,0] = False
  nobjs = len(set(np.unique(lab_gt)) - {0})
  total = iou[matching].sum()
  if partial_dataset:
    return total , nobjs
  else:
    return total / nobjs

def intersection_over_union(psg):
  rsum = np.sum(psg, 0, keepdims=True)
  csum = np.sum(psg, 1, keepdims=True)
  return psg / (rsum + csum - psg)

def matching_overlap(psg, fractions=(0.5,0.5)):
  """
  create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
  NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
  NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
  """
  afrac, bfrac = fractions
  m0 = psg / np.sum(psg, axis=1, keepdims=True)
  m1 = psg / np.sum(psg, axis=0, keepdims=True)
  m0 = m0 > afrac
  m1 = m1 > bfrac
  matching = m0 * m1
  matching = matching.astype('bool')
  return matching

def matching_iou(psg, fraction=0.5):
  iou = intersection_over_union(psg)
  matches = iou > 0.5
  matches[:,0] = False
  matches[0,:] = False
  return matches

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

@jit
def pixel_sharing_bipartite(img1, img2):
  """
  returns an ndarray representing a bipartite graph with pixel overlap count as the edge weight.
  img1 and img2 must be same shape, and label (int) images.
  """
  img1 = img1.astype(np.int64)
  img2 = img2.astype(np.int64)
  l1max = int(img1.max()+1)
  l2max = int(img2.max()+1)
  if img1.ndim==2:
      imgs = np.stack((img1, img2), axis=2)
      psg = np.zeros((l1max, l2max), dtype=np.uint32)
      a,b,c = imgs.shape
      for i in range(a):
          for j in range(b):
              psg[imgs[i,j,0], imgs[i,j,1]] += 1
      return psg
  elif img1.ndim==3:
      imgs = np.stack((img1, img2), axis=3)
      psg = np.zeros((l1max, l2max), dtype=np.uint32)
      a,b,c,d = imgs.shape
      for i in range(a):
          for j in range(b):
              for k in range(c):
                  psg[imgs[i,j,k,0], imgs[i,j,k,1]] += 1
      return psg

def denseQ(lab):
  return set(np.arange(lab.min(), lab.max()+1)) - set(np.unique(lab))

def matchingQ(matching):
  b0 = matching.dtype in [np.bool, np.uint8, np.uint16, np.uint32, np.uint64]
  b1 = np.sum(matching,0).max() == np.sum(matching,1).max() <= 1
  if not b0:
    raise TypeError("Matching should be bool or uint type.")
  return b1

def make_dense(lab):
  for l in denseQ(lab):
    lab[lab==lab.max()] = l

def maps_from_matching(matching):
  """
  matching between all nonzero ids
  """
  map1 = dict()
  map2 = dict()
  for i in range(matching.shape[0]):
    m = matching[i].argmax()
    if m > 0:
      map1[i] = m
      map2[m] = i
  return map1, map2

def precision(lab_gt, lab):
  """
  precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching
  """
  psg = segtools_simple.pixel_sharing_bipartite(lab_gt, lab)
  matching = segtools_simple.matching_iou(psg, fraction=0.5)
  assert matching.sum(0).max() < 2
  assert matching.sum(1).max() < 2
  n_gt  = len(set(np.unique(lab_gt)) - {0})
  n_hyp = len(set(np.unique(lab)) - {0})
  n_matched = matching.sum()
  precision = n_matched / (n_gt + n_hyp - n_matched)
  return precision

def matching_sets(lab_gt, lab):
  m1,m2 = matching_maps(lab_gt, lab)
  s1 = set(m1.keys())
  s2 = set(m2.keys())
  s1c = (set(np.unique(lab_gt)) - {0}) - s1
  s2c = (set(np.unique(lab))    - {0}) - s2
  return s1,s2,s1c,s2c

def matching_maps(lab_gt, lab):
  psg = segtools_simple.pixel_sharing_bipartite(lab_gt, lab)
  matching = segtools_simple.matching_iou(psg, fraction=0.5)
  assert matching.sum(0).max()<2
  assert matching.sum(1).max()<2
  map1, map2 = segtools_simple.maps_from_matching(matching)
  return map1, map2

def matching_masks(lab_gt, lab):
  s1,s2,s1c,s2c = matching_sets(lab_gt, lab)
  mask1  = lib.mask_labels(s1, lab_gt)
  mask2  = lib.mask_labels(s2, lab)
  mask1c = lib.mask_labels(s1c, lab_gt)
  mask2c = lib.mask_labels(s2c, lab)
  return mask1, mask2, mask1c, mask2c