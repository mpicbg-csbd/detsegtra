import numpy as np
from numba import jit
import networkx as nx


def compute_seg(hyp_gt, hyp_seg):
  psg = pixel_sharing_bipartite(hyp_gt, hyp_seg)
  seg_score = _seg_orig(psg)
  print("SEG", seg_score)
  matching = matching_overlap(psg)
  overlap_matching_score = matching_score(matching) # only compute this value for the printed side-effects
  return seg_score


def vtx_entropy(psg, axis=0):
  entropy = psg/psg.sum(axis=axis, keepdims=True)
  entropy = np.where(entropy!=0, entropy*np.log2(entropy), 0)
  entropy = -entropy.sum(axis=axis)
  return entropy
  # ent_map = {i:entropy[i] for i in range(len(entropy))}

def compute_seg_and_relabel(hyp_gt, hyp_seg):
  """
  takes two labeled ndarrays. 2d or 3d. pixel values are unique labels.
  computes seg_score and the discrete matching score.
  returns seg_score, inverse label mapping, and the 2nd array (hyp_seg) relabeled.
  """
  assert denseQ(hyp_gt)
  assert denseQ(hyp_seg)
  
  psg = pixel_sharing_bipartite(hyp_gt, hyp_seg)
  seg_entropy = vtx_entropy(psg, axis=0)
  gt_entropy  = vtx_entropy(psg, axis=1)

  seg_score = _seg_orig(psg)
  print("SEG", seg_score)
  matching  = matching_overlap(psg)
  overlap_matching_score = matching_score(matching) # only compute this value for the printed side-effects
  seg_map = mapping_from_matching(matching)
  res = {}
  res['seg'] = seg_score
  res['seg_map'] = seg_map
  res['ent_seg'] = seg_entropy
  res['ent_gt']  = gt_entropy
  # res['ent_map'] = ent_map
  res['seg_match_tuple'] = overlap_matching_score
  return res

def identify_membrane(psg):
  "only works with dense lableing"
  memlabel = np.argmax((psg!=0).sum(0))
  return memlabel

def psg2nx(psg):
  a,b = psg.shape
  tr  = np.zeros((b,b))
  bl  = np.zeros((a,a))
  top = np.concatenate([psg, tr], axis=0)
  bot = np.concatenate([bl, psg.T], axis=0)
  mat = np.concatenate([top, bot], axis=1)
  bipartite_nx = nx.from_numpy_matrix(mat)
  return bipartite_nx

@jit
def pixel_sharing_bipartite(img1, img2):
  """
  returns an ndarray representing a bipartite graph with pixel overlap count as the edge weight.
  img1 and img2 must be same shape, and label (uint) images.
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

def _seg_orig(psg):
  """
  calculate seg from pixel_sharing_bipartite
  seg is the average conditional-iou across ground truth cells
  conditional-iou gives zero if not in matching
  ----
  calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
  for a fraction > 0.5 matching. Any CIoU will be > 1/3. But there may be some
  IoU as low as 1/2 that don't match, and thus have CIoU = 0.
  """
  matching = matching_overlap(psg, fraction=0.5)
  iou = intersection_over_union(psg)
  conditional_iou = matching * iou
  seg = np.max(conditional_iou, axis=1)
  seg = np.mean(seg)
  return seg

def intersection_over_union(psg):
  rsum = np.sum(psg, 0, keepdims=True)
  csum = np.sum(psg, 1, keepdims=True)
  return psg / (rsum + csum - psg)

def matching_overlap(psg, fraction=0.5):
  """
  create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
  NOTE: a true matching is only gauranteed for fraction > 0.5. Otherwise some cells might have deg=2 or more.
  NOTE: doesn't break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
  """
  matc = psg / np.sum(psg, axis=1, keepdims=True)
  matr = psg / np.sum(psg, axis=0, keepdims=True)
  matc50 = matc > fraction
  matr50 = matr > fraction
  matching = matc50 * matr50
  matching = matching.astype('uint8')
  return matching

def matching_score(matching):
  print("{} matches out of {} GT objects and {} predicted objects.".format(matching.sum(), matching.shape[0], matching.shape[1]))
  return (matching.sum(), matching.shape[0], matching.shape[1])

@jit
def relabel_img(img, mapping_array=None):
  """
  Permute the labels on a labeled image according to `mapping_array`, if `mapping_array` not given
  then permute them randomly.
  Returns a copy of `img`.
  """
  res = img.copy()
  if mapping_array is None:
      mapping_array = np.arange(img.max()+1)
      np.random.shuffle(mapping_array)
  if img.ndim==2:
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i,j] = mapping_array[img[i,j]]
  elif img.ndim==3:
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
          for k in range(img.shape[2]):
              res[i,j,k] = mapping_array[img[i,j,k]]
  return res

def mapping_from_matching(matching):
  """
  return the mapping from seg labels (dim 1) to gt labels (dim 0).
  if a label has no match, then it gets a new, large, unique label.
  intersection of new labels with gt labels is only the label set of matches.
  """
  mapping = {}
  l_max = matching.shape[0]
  for i in range(matching.shape[1]):
    k = np.argmax(matching[:,i])
    if k == 0 and matching[k,i]==0:
      k = l_max + 1
      l_max += 1
    mapping[i] = k
  return mapping

def condense_labels(lab):
  """
  takes non-dense array and returns the dense version and the inverse mapping of new labels to old.
  """
  mapping = {}
  uniq = np.unique(lab)
  newlabels = np.arange(len(uniq))
  for k,v in zip(uniq, newlabels):
    mapping[k] = v
  newlab, mapinv = apply_mapping(lab, mapping)
  return newlab, mapinv

def apply_mapping(lab, mapping):
  maxlabel = max(mapping.keys())
  maparray = np.zeros(int(maxlabel+1), dtype=np.uint32)
  mapinv = {}
  for k,v in mapping.items():
    maparray[k] = v
    mapinv[v] = k
  lab2 = relabel_img(lab, maparray)
  return lab2, mapinv

def denseQ(lab):
  uniq = np.unique(lab)
  maxv = lab.max()
  b1 = len(uniq) == maxv+1
  if not b1:
    return False
  b2 = (np.unique(lab)==np.arange(maxv+1)).all()
  if not b2:
    return False
  return True
