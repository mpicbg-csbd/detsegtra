from pykdtree.kdtree import KDTree as pyKDTree
import numpy as np
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment
import ipdb


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

def old_match_to_scores(match):
  res = SimpleNamespace()
  res.n_matched = match[0]
  res.n_proposed = match[1]
  res.n_gt = match[2]
  res.precision  = res.n_matched / res.n_proposed
  res.recall     = res.n_matched / res.n_gt
  res.f1         = 2*res.n_matched / (res.n_proposed + res.n_gt)
  return res

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

def match_unambiguous_nearestNeib(_pts_gt,_pts_yp,dub=10,scale=[1,1,1]):
  """
  pts_gt is ground truth. pts_yp as predictions. this function is not symmetric!
  we return binary masks for pts_gt and pts_yp where masked elements are matched.
  we also return a mapping from gt2yp and yp2gt matching indices.
  
  We can obtain a matching between points in many different ways.
  The old way was to only compute the function from gt to yp.
  Thus an individual yp may appear zero or more times.
  
  BUG: we never established a true matching, just the score.
  Our scheme was such that all gt points within `dub` distance of a yp were considered matched, even if that match was not unique.
  This makes sense if the matching region associated with a nucleus is _uniquely_ claimed by that nucleus (regions don't overlap).
  If regions _do_ overlap, then this second criterion is active (nucleus center is nearest neib of proposed point).
  We could solve an assignment problem with Hungarian matching to enable even more flexible matching.
  This is only necessary if we have overlapping regions, and it might be possible that proposed point X matches to gt point Y1 even though it is closer to Y2.

  Tue Apr 13 13:35:02 2021
  Return nan for precision|recall|f1 when number of objects is zero
  """

  res = SimpleNamespace()

  def _final_scores(n_m,n_p,n_gt):
    with np.errstate(divide='ignore',invalid='ignore'):
      precision = np.divide(n_m  ,  n_p)
      recall    = np.divide(n_m  ,  n_gt)
      f1        = np.divide(2*n_m,  (n_p + n_gt))

    res = SimpleNamespace()
    res.n_matched = n_m
    res.n_proposed = n_p
    res.n_gt = n_gt
    res.precision = precision
    res.f1 = f1
    res.recall = recall
    return res

  if len(_pts_gt)==0 or len(_pts_yp)==0:
    n_matched  = 0
    n_proposed = len(_pts_yp)
    n_gt       = len(_pts_gt)
    return _final_scores(n_matched,n_proposed,n_gt)

  pts_gt = np.array(_pts_gt) * scale ## for matching in anisotropic spaces
  pts_yp = np.array(_pts_yp) * scale ## for matching in anisotropic spaces

  kdt = pyKDTree(pts_yp)
  gt2yp_dists, gt2yp = kdt.query(pts_gt, k=1, distance_upper_bound=dub)
  kdt = pyKDTree(pts_gt)
  yp2gt_dists, yp2gt = kdt.query(pts_yp, k=1, distance_upper_bound=dub)

  N = len(pts_gt)
  inds = np.arange(N)
  ## must extend yp2gt with N for gt points whose nearest neib is beyond dub
  res.gt_matched_mask = np.r_[yp2gt,N][gt2yp]==inds
  N = len(pts_yp)
  inds = np.arange(N)
  res.yp_matched_mask = np.r_[gt2yp,N][yp2gt]==inds

  assert res.gt_matched_mask.sum() == res.yp_matched_mask.sum()
  res.gt2yp = gt2yp
  res.yp2gt = yp2gt
  res.pts_gt = _pts_gt ## take normal points, not rescaled!
  res.pts_yp = _pts_yp ## take normal points, not rescaled!

  res.n_matched  = res.gt_matched_mask.sum()
  res.n_proposed = len(pts_yp)
  res.n_gt       = len(pts_gt)
  res.precision  = res.n_matched / res.n_proposed
  res.recall     = res.n_matched / res.n_gt
  res.f1         = 2*res.n_matched / (res.n_proposed + res.n_gt)

  return res

def test_matching():
  x = np.random.rand(10)[...,None]*100
  y = np.random.rand(14)[...,None]*100
  sym = match_unambiguous_nearestNeib(x,y,dub=1,scale=1)
  import matplotlib.pyplot as plt
  plt.plot(x, np.zeros(len(x)), 'o')
  plt.plot(y, np.ones(len(y)), 'o')
  for i in np.arange(len(x))[sym.gt_matched_mask]:
    plt.plot([x[i],y[sym.gt2yp[i]]],[0,1],'k')


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

def listOfMatches_to_Scores(listOfMatches):
  """
  works for any matching that gives n_gt, n_proposed, and n_matched.
  """
  res = SimpleNamespace()
  res.n_gt = sum([h.n_gt for h in listOfMatches])
  res.n_proposed = sum([h.n_proposed for h in listOfMatches])
  res.n_matched  = sum([h.n_matched for h in listOfMatches])
  res.precision  = res.n_matched / res.n_proposed
  res.recall     = res.n_matched / res.n_gt
  res.f1         = 2*res.n_matched / (res.n_proposed + res.n_gt)
  return res





