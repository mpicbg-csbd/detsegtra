from pykdtree.kdtree import KDTree as pyKDTree
import numpy as np
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment

def match_points_single(pts_gt,pts_yp,dub=10):
  "pts_gt is ground truth. pts_yp as predictions. this function is not symmetric!"
  pts_gt = np.array(pts_gt)
  pts_yp = np.array(pts_yp)
  if 0 in pts_gt.shape: return 0,len(pts_yp),len(pts_gt)
  if 0 in pts_yp.shape: return 0,len(pts_yp),len(pts_gt)
  # print(pts_gt.shape, pts_yp.shape)

  kdt = pyKDTree(pts_yp)
  dists, inds = kdt.query(pts_gt, k=1, distance_upper_bound=dub)
  matched,counts = np.unique(inds[inds<len(pts_yp)], return_counts=True)
  return len(matched), len(pts_yp), len(pts_gt)

def matches2scores(matches):
  """
  matches is an Nx3 array with (n_matched, n_proposed, n_target) semantics.
  here we perform mean-then-divide to compute scores. As opposed to divide-then-mean.
  """
  d = SimpleNamespace()
  d.f1          = 2*matches[:,0].sum() / np.maximum(matches[:,[1,2]].sum(),1)
  d.precision   =   matches[:,0].sum() / np.maximum(matches[:,1].sum(),1)
  d.recall      =   matches[:,0].sum() / np.maximum(matches[:,2].sum(),1)
  d.f1_2        = (2*matches[:,0] / np.maximum(matches[:,[1,2]].sum(1),1)).mean()
  d.precision_2 = (  matches[:,0] / np.maximum(matches[:,1],1)).mean()
  d.recall_2    = (  matches[:,0] / np.maximum(matches[:,2],1)).mean()

  return d



def match_unambiguous_nearestNeib(pts_gt,pts_yp,dub=10,scale=[1,1,1]):
  """
  pts_gt is ground truth. pts_yp as predictions. this function is not symmetric!
  we return binary masks for pts_gt and pts_yp where masked elements are matched.
  we also return a mapping from gt2yp and yp2gt matching indices.
  
  We can obtain a matching between points in many different ways.
  The old way was to only compute the function from gt to yp.
  Thus an individual yp may appear zero or more times.
  
  BUG: we never established a true matching, just the score. 
  Our scheme was such that all gt points within `dub` distance of a yp were considered matched, even if that match was not unique.
  This is probably not correct...
  """

  res = SimpleNamespace()

  pts_gt = np.array(pts_gt) * scale ## for matching in anisotropic spaces
  pts_yp = np.array(pts_yp) * scale ## for matching in anisotropic spaces

  if 0 in pts_gt.shape or 0 in pts_yp.shape:
    res.totals = 0,len(pts_yp),len(pts_gt)
    res.dists  = np.zeros(len(pts_gt))-1
    res.gt2yp  = np.zeros(len(pts_gt))-1
    return res

  kdt = pyKDTree(pts_yp)
  gt2yp_dists, gt2yp = kdt.query(pts_gt, k=1, distance_upper_bound=dub)
  gt2yp_mask = gt2yp<len(pts_yp)

  kdt = pyKDTree(pts_gt)
  yp2gt_dists, yp2gt = kdt.query(pts_yp, k=1, distance_upper_bound=dub)
  yp2gt_mask = yp2gt<len(pts_gt)


  ## matches are objects where the connections form a cycle. i.e. f:x->y, g:y->x and x_i = g(f(x_i))
  res.gt_matches = np.arange(len(pts_gt))[gt2yp_mask] == yp2gt[gt2yp[gt2yp_mask]]
  res.yp_matches = np.arange(len(pts_yp))[yp2gt_mask] == gt2yp[yp2gt[yp2gt_mask]]

  assert res.gt_matches.sum() == res.yp_matches.sum()

  # res.matched, counts = np.unique(res.gt2yp[res.gt2yp_mask], return_counts=True)
  # res.totals = len(matched), len(pts_yp), len(pts_gt)
  res.n_matched  = res.gt_matches.sum()
  res.n_proposed = len(pts_yp)
  res.n_gt       = len(pts_gt)
  res.precision  = res.n_matched / res.n_proposed
  res.recall     = res.n_matched / res.n_gt
  res.f1         = 2*res.n_matched / (res.n_proposed + res.n_gt)

  return res

def hungarian_matching(x,y,scale=[1,1,1]):
  """
  matching that minimizes sum of costs (in this case euclidean distance).
  the 
  """
  hun = SimpleNamespace()
  hun.cost = np.zeros((len(x), len(y)))
  for i,c in enumerate(x):
    for j,d in enumerate(y):
      hun.cost[i,j] = np.linalg.norm((c-d)*scale)
  hun.lsa = linear_sum_assignment(hun.cost)
  return hun

def score_hungarian(hun,dub=2.5):
  mtx = np.zeros(hun.cost.shape)

  mtx[hun.lsa[0],hun.lsa[1]] = 1
  mtx[hun.cost > dub] = 0
  hun.mtx = mtx
  hun.x_mask = mtx.sum(1).astype(np.bool)
  hun.y_mask = mtx.sum(0).astype(np.bool)
  hun.x2y    = mtx.argmax(1); hun.x2y[~hun.x_mask] = -1
  hun.y2x    = mtx.argmax(0); hun.y2x[~hun.y_mask] = -1
  hun.n_matched  = mtx.sum()
  hun.n_gt       = mtx.shape[0]
  hun.n_proposed = mtx.shape[1]
  hun.precision  = hun.n_matched / hun.n_proposed
  hun.recall     = hun.n_matched / hun.n_gt
  hun.f1         = 2*hun.n_matched / (hun.n_proposed + hun.n_gt)
  return hun

def listOfMatches_to_Scores(hungs):
  res = SimpleNamespace()
  res.n_gt = sum([h.n_gt for h in hungs])
  res.n_proposed = sum([h.n_proposed for h in hungs])
  res.n_matched  = sum([h.n_matched for h in hungs])
  res.precision  = res.n_matched / res.n_proposed
  res.recall     = res.n_matched / res.n_gt
  res.f1         = 2*res.n_matched / (res.n_proposed + res.n_gt)
  return res




def listOfMatches2Scores(matches):
  """
  matches is an Nx3 array with (n_matched, n_proposed, n_target) semantics.
  """
  matches = np.array(matches)
  d = SimpleNamespace()
  d.f1          = 2*matches[:,0].sum() / np.maximum(matches[:,[1,2]].sum(),1)
  d.precision   =   matches[:,0].sum() / np.maximum(matches[:,1].sum(),1)
  d.recall      =   matches[:,0].sum() / np.maximum(matches[:,2].sum(),1)
  d.f1_2        = (2*matches[:,0] / np.maximum(matches[:,[1,2]].sum(1),1)).mean()
  d.precision_2 = (  matches[:,0] / np.maximum(matches[:,1],1)).mean()
  d.recall_2    = (  matches[:,0] / np.maximum(matches[:,2],1)).mean()
  return d
