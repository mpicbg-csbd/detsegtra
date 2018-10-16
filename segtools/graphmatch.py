from pykdtree.kdtree import KDTree as pyKDTree
import networkx as nx
import numpy as np
from numba import jit
from skimage import measure
from collections import Counter

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

def connect_points_symmetric(x, y, **kwargs):
  """
  Build a bipartite from two point clouds with edges between
  all points within cutoff radius.
  coords are (n,m) arrays. each arr[i,:] is a vec in arbitrary feature space.
  distance between points is euclidean.
  """
  map_x2y = x2y_labelmap(x, y, **kwargs)
  map_y2x = x2y_labelmap(y, x, **kwargs)
  g1 = nx.from_dict_of_lists(map_x2y)
  g2 = nx.from_dict_of_lists(map_y2x)
  g3 = nx.compose(g1, g2)
  return g3

def connect_points_digraph_symmetric(x, y, reverse=True, **kwargs):
  map_x2y = x2y_labelmap(x, y, **kwargs)
  map_y2x = x2y_labelmap(y, x, **kwargs)
  g1 = nx.from_dict_of_lists(map_x2y, nx.DiGraph())
  g2 = nx.from_dict_of_lists(map_y2x, nx.DiGraph())
  if reverse:
    g2 = g2.reverse()
  g3 = nx.compose(g1, g2)
  return g3

def connect_points_digraph(x, y, **kwargs):
  map_x2y = x2y_labelmap(x, y, **kwargs)
  g1 = nx.from_dict_of_lists(map_x2y, nx.DiGraph())
  return g1

def kdmatch(x,y,k=7,dub=100):
  kdt = pyKDTree(y)
  dists, inds = kdt.query(x, k=k, distance_upper_bound=dub)
  inds = inds.reshape((-1,k))
  return inds

def x2y_labelmap(x, y, lx='x', ly='y', labels_x=None, labels_y=None, **kwargs):
  if labels_x is None:
    labels_x = np.arange(x.shape[0])
  if labels_y is None:
    labels_y = np.arange(y.shape[0])

  indices_x2y = kdmatch(x, y, **kwargs)
  labels_x  = np.concatenate([labels_x, [-1]])
  labels_y  = np.concatenate([labels_y, [-1]])
  labels_x2y = {}
  for i,vs in enumerate(indices_x2y):
    labels_x2y[(lx, labels_x[i])] = [(ly, v) for v in labels_y[vs] if v!=-1]
  return labels_x2y


def symmetric_unique_nolabel_connection(pts0,pts1,**kwargs):
  inds_x2y = kdmatch(pts0,pts1,k=1,**kwargs)
  matched_ys = pts1[inds_x2y[inds_x2y<len(pts1)]]
  inds_y2x = kdmatch(pts1,pts0,k=1,**kwargs)
  matched_xs = pts0[inds_y2x[inds_y2x<len(pts0)]]




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

def unique_matching(bipartite):
  match = {}
  deg = bipartite.degree()
  for v,u in bipartite.edges():
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

## stats

def match_stats(nx_bipartite, matching, from_x='gt_', to_y='seg_'):
  n_gt            = len([n for (t,n) in nx_bipartite.node if t==from_x])
  n_seg           = len([n for (t,n) in nx_bipartite.node if t==to_y])
  n_matched       = len(matching.keys())//2
  n_gt_unmatched  = n_gt - n_matched
  n_seg_unmatched = n_seg - n_matched
  n_unmatched     = n_gt_unmatched + n_seg_unmatched

  degrees_gt  = [deg for k,deg in nx_bipartite.degree() if k[0]=='gt_']
  degdist_gt  = Counter(degrees_gt)
  degrees_seg = [deg for k,deg in nx_bipartite.degree() if k[0]=='seg_']
  degdist_seg = Counter(degrees_seg)

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





