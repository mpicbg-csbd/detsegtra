from pykdtree.kdtree import KDTree as pyKDTree
import networkx as nx
import numpy as np
from numba import jit
from skimage import measure
from .loc_utils import list2dist

"""
This module works with networkx-based bipartite graphs and matchings
"""
    
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

## ---- Conversion from matrix to networkx [bipartite]

def matrix2bipartite(matrix, labs1, labs2):
  "labs1 and labs2 are label sets."  
  m = {}
  for l1 in labs1:
      m[('gt_', l1)] = [('seg_', l2) for l2 in labs2 if matrix[l1,l2] == 1]
  g = nx.from_dict_of_lists(m)
  return g

def matrix2weightedbipartite(matrix, labs1, labs2, w='psg'):
  "labs1 and labs2 are label sets. w is name of weight."
  m = {}
  for l1 in labs1:
      m[('gt_', l1)] = {('seg_', l2) : {w : matrix[l1,l2]} for l2 in labs2 if matrix[l1,l2] > 0}
  g = nx.from_dict_of_dicts(m)
  return g

def psg2nx(psg):
  a,b = psg.shape
  tr = np.zeros((b,b))
  bl  = np.zeros((a,a))
  top = np.concatenate([psg, tr], axis=0)
  bot = np.concatenate([bl, psg.T], axis=0)
  mat = np.concatenate([top, bot], axis=1)
  bipartite_nx = nx.from_numpy_matrix(mat)
  return bipartite_nx

## ---- useful bipartite graphs ----

def centerpoint_in_seg_bipartite(hyp_gt, hyp_seg, coords_gt, coords_seg, gtlabels, seglabels):
  a,b = int(hyp_gt.max()), int(hyp_seg.max())
  bipartite = np.zeros((a+1,b+1), dtype=np.int)

  inds = coords_gt.astype(np.int)
  if inds.shape!=(0,):
    gt2seglabel = hyp_seg[tuple(inds.T)]
    bipartite[gtlabels, gt2seglabel] = 1

  inds = coords_seg.astype(np.int)
  if inds.shape!=(0,):
    seg2gtlabel = hyp_gt[tuple(inds.T)]
    bipartite[seg2gtlabel, seglabels] = 1
  
  bipartite = matrix2bip(bipartite, gtlabels, seglabels)
  return bipartite

def centroid_bipartite(coords_gt, coords_seg, l1='gt_', l2='seg_', labels_gt=None, labels_seg=None, k=7, dub=10):
  """
  Build a bipartite from two point clouds with edges between
  all points within cutoff radius.
  coords are (n,m) arrays. each arr[i,:] is a vec in arbitrary feature space.
  distance between points is euclidean.
  """

  def kdmatch(x,y):
    kdt = pyKDTree(y)
    dists, inds = kdt.query(x, k=k, distance_upper_bound=dub)
    return inds

  indices_gt2seg = kdmatch(coords_gt, coords_seg)
  indices_seg2gt = kdmatch(coords_seg, coords_gt)

  if labels_gt is None:
    labels_gt = np.arange(coords_gt.shape[0])
  if labels_seg is None:
    labels_seg = np.arange(coords_seg.shape[0])

  labels_gt  = np.concatenate([labels_gt, [-1]])
  labels_seg = np.concatenate([labels_seg, [-1]])

  labels_gt2seg = {}
  labels_seg2gt = {}
  for i,vs in enumerate(indices_gt2seg):
    labels_gt2seg[(l1, labels_gt[i])] = [(l2, v) for v in labels_seg[vs] if v!=-1]
  for i,vs in enumerate(indices_seg2gt):
    labels_seg2gt[(l2, labels_seg[i])] = [(l1, v) for v in labels_gt[vs] if v!=-1]

  g1 = nx.from_dict_of_lists(labels_gt2seg)
  g2 = nx.from_dict_of_lists(labels_seg2gt)
  g3 = nx.compose(g1, g2)
  
  return g3

@DeprecationWarning
def psg_bipartite(hyp_gt, hyp_seg):
  psg = pixel_sharing_bipartite(hyp_gt, hyp_seg)
  # bipartite = array2weightedbip(psg, labels_gt, labels_seg, w='psg')
  bipartite = {}
  areas = {}
  labels_gt = np.arange(hyp_gt.max() + 1, dtype=np.int)
  labels_seg = np.arange(hyp_seg.max() + 1, dtype=np.int)
  for l1 in labels_gt:
      a = psg[l1,:].sum()
      if a > 0:
        areas[('gt_', l1)] = a
        bipartite[('gt_', l1)] = {('seg_', l2) : {'overlap' : psg[l1,l2]} for l2 in labels_seg if psg[l1,l2] > 0}
  for l2 in labels_seg:
      a = psg[:,l2].sum()
      if a > 0:
        areas[('seg_', l2)] = a
  g = nx.from_dict_of_dicts(bipartite)
  nx.set_node_attributes(g, 'area', areas)
  return g

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

def weighted_bipartite2matching(bipartite, func):
  match = {}
  for v,d in bipartite.node.items():
    a1 = d['area']
    for e,d in bipartite.edge[v].items():
      ov = d['overlap']
      a2 = bipartite.node[e]['area']
      d['iou'] = ov/(a1 + a2 - ov)
      if ov/a1 >= 0.5 and ov/a2 >= 0.5:
        match[v] = e
  return match

## maximum matching with nx.bipartite.maximum_matching

def compare_nhls(nhl_gt, nhl_seg):
  coords_gt, labels_gt = get_centroids(hyp_gt)

  bipartite = centroid_bipartite(coords_gt, coords_seg, labels_gt, labels_seg, dub=dub)
  # matching  = nx.bipartite.maximum_matching(bipartite)
  matching  = unique_matching(bipartite)
  stats     = match_stats(bipartite, matching, from_x='gt_', to_y='seg_')
  mapping   = match2map(bipartite, matching, from_x='gt_', to_y='seg_')

def stats_center2seg_matching(hyp_gt, hyp_seg):
  coords_gt, labels_gt = get_centroids(hyp_gt)
  coords_seg, labels_seg = get_centroids(hyp_seg)
  bipartite = centroid2seg_bipartite(hyp_gt, hyp_seg, coords_gt, coords_seg, labels_gt, labels_seg)
  matching  = unique_matching(bipartite)
  results = make_results(bipartite, matching)
  mapping = results['map']
  return results

def make_results(bipartite, matching):
  stats     = match_stats(bipartite, matching, from_x='gt_', to_y='seg_')
  mapping   = match2map(bipartite, matching, from_x='gt_', to_y='seg_')

  results = {'stats':stats, 'bip':bipartite, 'match':matching, 'map':mapping}
  results['matched_seg_set'] = {k[1] for k in matching.keys() if k[0]=='seg_'}
  results['matched_gt_set']  = {k[1] for k in matching.keys() if k[0]=='gt_'}
  return results

def stats_seg_matching(hyp_gt, hyp_seg):
  bipartite = psg_bipartite(hyp_gt, hyp_seg)
  matching  = seg_matching(bipartite)
  results = make_results(bipartite, matching)
  mapping = results['map']
  return results





