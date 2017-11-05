doc = """
# Module for Label images

We want to think about some tests, and perhaps a spec for labeled images. They
should be uint type?
The labeled images that Carine made have zero-valued membranes which separate the
>= 2 valued cells and the 1-valued background. But we don't *need* to have this
boundary layer, and it probably also shouldn't count towards our cell-matching score. (Neither should the background?).
"""

from pykdtree.kdtree import KDTree as pyKDTree
import networkx as nx
import numpy as np
import skimage.io as io
import colorsys
from numba import jit
from skimage import measure

from src import lib

import sys
sys.path.insert(0, "/Users/colemanbroaddus/Desktop/Projects/nucleipix/")

# or get it from scipy.ndimage.morphology import generate_binary_structure
structure = [[1,1,1], [1,1,1], [1,1,1]] # this is the structure that was used by Benoit & Carine!

## JUST COLORING STUFF

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


# MISC

@DeprecationWarning
@jit
def permute_img(img, perm=None):
    """
    Permute the labels on a labeled image according to `perm`, if `perm` not given
    then permute them randomly.
    Returns a copy of `img`.
    """
    if not perm:
        perm = np.arange(img.max()+1)
        np.random.shuffle(perm)
    res = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i,j] = perm[img[i,j]]
    return res

@DeprecationWarning
def permutation_from_matching(matching):
    ar = np.arange(matching.shape[0])
    p1 = np.argmax(matching, axis=1)
    # p1 is *almost* the permutation we want...
    # what do we do if matching[i,j]=1, but matching[j,:] = all zeros ?
    # which label do we give to j in perm? a new, biggest label?
    # yep, that's what we'll do...
    # this way the intersection of labels in the two images are just the ones in the matching!
    perm = np.where(matching[ar,p1]!=0, p1, -1)
    s = perm[perm==-1].shape[0]
    perm[perm==-1] = np.arange(s)+perm.max()+1
    return perm

# @jit
# def matching_from_permutation(perm):
#     matching = np.zeros(perm[0])

# Segmentation LOSSES, ERRORS, SCORES, MATCHINGS, GRAPHS

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


@DeprecationWarning
def psg2nx(psg):
  a,b = psg.shape
  tr = np.zeros((b,b))
  bl  = np.zeros((a,a))
  top = np.concatenate([psg, tr], axis=0)
  bot = np.concatenate([bl, psg.T], axis=0)
  mat = np.concatenate([top, bot], axis=1)
  bipartite_nx = nx.from_numpy_matrix(mat)
  return bipartite_nx

def centerpoint_in_seg_bipartite(hyp_gt, hyp_seg, coords_gt, coords_seg, gtlabels, seglabels):
    inds = coords_gt.astype(np.int)
    gt2seglabel = hyp_seg[tuple(inds.T)]
    inds = coords_seg.astype(np.int)
    seg2gtlabel = hyp_gt[tuple(inds.T)]
    vals,cts = np.unique(gt2seglabel, return_counts=True)
    a,b = int(hyp_gt.max()), int(hyp_seg.max())
    bipartite = np.zeros((a+1,b+1), dtype=np.int)
    bipartite[gtlabels, gt2seglabel] = 1
    bipartite[seg2gtlabel, seglabels] = 1
    return bipartite

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

def matching_max(psg):
    """
    matching based on most overlapping pixels
    """
    rowmax = np.argmax(psg, axis=0)
    colmax = np.argmax(psg, axis=1)
    starting_index = np.arange(len(rowmax))
    equal_matches = colmax[rowmax[starting_index]]==starting_index
    rm, cm = rowmax[equal_matches], colmax[rowmax[equal_matches]]
    matching = np.zeros_like(psg)
    matching[rm, cm] = 1
    return matching

def intersection_over_union(psg):
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)

def seg(psg):
    """
    calculate seg from pixel_sharing_bipartite
    seg is the average conditional-iou across ground truth cells
    conditional-iou gives zero if not in matching
    ----
    calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
    for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
    IoU as low as 1/2 that don't match, and thus have CIoU = 0.
    """
    matching = matching_overlap(psg, fraction=0.5)
    iou = intersection_over_union(psg)
    conditional_iou = matching * iou
    seg = np.max(conditional_iou, axis=1)
    seg = np.mean(seg)
    return seg

def matching_score(matching):
  "matching is numpy matching, i.e. must satisfy `is_matching_matrix`"  
  print("{} matches out of {} GT objects and {} predicted objects.".format(matching.sum(), matching.shape[0], matching.shape[1]))
  return (matching.sum(), matching.shape[0], matching.shape[1])
    
def is_matching_matrix(matching):
    assert matching.dtype in [np.bool, np.uint8, np.uint16, np.uint32, np.uint64]
    assert np.sum(matching,0).max() == np.sum(matching,1).max() <= 1
    return True


## ---- Network X ----



## ---- Utility ----

def list2dist(lst):
    val, cts = np.unique(lst, return_counts=True)
    dist = dict(zip(val, cts))
    return dist

def get_centroids(hyp):
    rps = measure.regionprops(hyp)
    coords = [np.mean(rp.coords, axis=0) for rp in rps]
    coords = np.array(coords)
    labels = [rp['label'] for rp in rps]
    labels = np.array(labels)
    return coords, labels

def match2map(bipartite, matching, from_x='gt_', to_y='seg_'):
  """
  build mapping from y labels to x labels.
  """
  left_labels = [n for (t,n) in bipartite.node if t==from_x]
  right_nodes = [(t,n) for (t,n) in bipartite.node if t==to_y]

  mapping = {}
  l_max = max(left_labels)
  for k in right_nodes:
    l = matching.get(k, -1)
    if l==-1:
        l=l_max+1
        l_max += 1
    else:
        l = l[1]
    mapping[k[1]] = l
  return mapping

def apply_mapping(lab, mapping):
  maxlabel = max(mapping.keys())
  maparray = np.zeros(maxlabel+1, dtype=np.uint32)
  mapinv = {}
  for k,v in mapping.items():
    maparray[k] = v
    mapinv[v] = k
  # lab2 = relabel_img(lab, maparray)
  lab2 = maparray[lab.flat].reshape(lab.shape)
  return lab2, mapinv

## ---- Conversion from matrix to networkx [bipartite]

def matrix2bip(bipartite, labs1, labs2):
    m = {}
    for l1 in labs1:
        m[('gt_', l1)] = [('seg_', l2) for l2 in labs2 if bipartite[l1,l2] == 1]
    g = nx.from_dict_of_lists(m)
    return g

def array2weightedbip(psg, labs1, labs2, w='psg'):
    m = {}
    for l1 in labs1:
        m[('gt_', l1)] = {('seg_', l2) : {w : psg[l1,l2]} for l2 in labs2 if psg[l1,l2] > 0}
    g = nx.from_dict_of_dicts(m)
    return g

## ---- useful bipartite graphs ----

def centroid_bipartite(coords_gt, coords_seg, labels_gt, labels_seg, dub=10):
  """
  Find the maximum bipartite matching between to point clouds with edges between
  all points within cutoff radius.
  Returns indices of matched points from `coords_seg` as well as summary of matching stats.
  """

  def kdmatch(x,y):
    kdt = pyKDTree(y)
    dists, inds = kdt.query(x, k=7, dub=dub)
    return inds

  indices_gt2seg = kdmatch(coords_gt, coords_seg)
  indices_seg2gt = kdmatch(coords_seg, coords_gt)

  labels_gt  = np.concatenate([labels_gt, [-1]])
  labels_seg = np.concatenate([labels_seg, [-1]])

  labels_gt2seg = {}
  labels_seg2gt = {}
  for i,vs in enumerate(indices_gt2seg):
    labels_gt2seg[('gt_', labels_gt[i])] = [('seg_', v) for v in labels_seg[vs] if v!=-1]
  for i,vs in enumerate(indices_seg2gt):
    labels_seg2gt[('seg_', labels_seg[i])] = [('gt_', v) for v in labels_gt[vs] if v!=-1]

  g1 = nx.from_dict_of_lists(labels_gt2seg)
  g2 = nx.from_dict_of_lists(labels_seg2gt)
  g3 = nx.compose(g1, g2)
  
  return g3

def centroid2seg_bipartite(hyp_gt, hyp_seg, coords_gt, coords_seg, labels_gt, labels_seg):
    bipartitearray = centerpoint_in_seg_bipartite(hyp_gt, hyp_seg, coords_gt, coords_seg, labels_gt, labels_seg)
    bipartite = matrix2bip(bipartitearray, labels_gt, labels_seg)
    return bipartite

def psg_bipartite(hyp_gt, hyp_seg, labels_gt, labels_seg):
    psg = pixel_sharing_bipartite(hyp_gt, hyp_seg)
    bipartite = array2weightedbip(psg, labels_gt, labels_seg, w='psg')
    return bipartite

## ---- matchings from bipartites ----

def match_stats(nx_bipartite, matching, from_x='gt_', to_y='seg_'):
  n_gt            = len([n for (t,n) in nx_bipartite.node if t==from_x])
  n_seg           = len([n for (t,n) in nx_bipartite.node if t==to_y])
  n_matched       = len(matching.keys())//2
  n_gt_unmatched  = n_gt - n_matched
  n_seg_unmatched = n_seg - n_matched
  n_unmatched     = n_gt_unmatched + n_seg_unmatched

  degrees_gt  = [deg for k,deg in nx_bipartite.degree() if k[0]=='gt_']
  degdist_gt  = list2dist(degrees_gt)
  degrees_seg = [deg for k,deg in nx_bipartite.degree() if k[0]=='seg_']
  degdist_seg = list2dist(degrees_seg)

  summary = {'n_cells'  : n_gt,
             'n_segs'   : n_seg,
             'n_match'  : n_matched,
             'n_un'     : n_unmatched,
             'n_gt_un'  : n_gt_unmatched,
             'n_seg_un' : n_seg_unmatched,
             'ratio_2'  : n_unmatched / n_gt,
             'ratio_1'  : n_unmatched / (n_matched+1),
             'dd_gt'    : degdist_gt,
             'dd_seg'   : degdist_seg}
  return summary

def unique_matching(bipartite):
    match = {}
    deg = bipartite.degree()
    for v,u in bipartite.edges_iter():
        if deg[v]==deg[u]==1:
            match[v]=u
            match[u]=v
    return match

## maximum weighted matching

## seg matching (use matrix2bip and 

## maximum matching with nx.bipartite.maximum_matching

## ---- useful functions on bipartites ----

def entropy(bipartite):
  "stub"
  deg = bipartite.deegree()
  print(deg)


def centroid_matching(hyp_gt, hyp_seg, dub=5):
  coords_gt, labels_gt = get_centroids(hyp_gt)
  coords_seg, labels_seg = get_centroids(hyp_seg)

  # bipartite = psg_bipartite(hyp_gt, hyp_seg, labels_gt, labels_seg)
  bipartite = centroid2seg_bipartite(hyp_gt, hyp_seg, coords_gt, coords_seg, labels_gt, labels_seg)  
  # bipartite = centroid_bipartite(coords_gt, coords_seg, labels_gt, labels_seg, dub=dub)

  # matching  = nx.bipartite.maximum_matching(bipartite)
  matching  = unique_matching(bipartite)
  stats     = match_stats(bipartite, matching, from_x='gt_', to_y='seg_')
  mapping   = match2map(bipartite, matching, from_x='gt_', to_y='seg_')
  
  hyp_seg_relabel, mapinv = apply_mapping(hyp_seg, mapping)

  results = {**stats, 'img':hyp_seg_relabel, 'bip':bipartite, 'match':matching, 'map':mapping}
  results['matched_seg_set'] = {k[1] for k in matching.keys() if k[0]=='seg_'}
  results['matched_gt_set']  = {k[1] for k in matching.keys() if k[0]=='gt_'}

  return results




def denseQ(lab):
  assert (np.unique(lab)==np.arange(lab.max()+1)).all()





### ------ DeprecationWarning below this line------

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







