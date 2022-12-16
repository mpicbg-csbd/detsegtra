## From Trackmeasure

import numpy as np
from . import point_matcher
import networkx as nx
import ipdb

from types import SimpleNamespace


def subsample_graph(tb,subsample=2):
  edges = []
  nodes = np.array(tb.nodes)
  tmax = nodes[:,0].max()
  newnodes = []

  for t in range(0,tmax+1,subsample):
    tnodes = nodes[nodes[:,0]==t]
    for n in tnodes:
      newnodes.append(n)
      n = tuple(n)
      cn = n    ## current node
      count = 0 ## how many parents have we climbed
      while True:
        l = list(tb.pred[cn])
        if len(l)==0: break
        elif (count==subsample-1):
          edges.append((l[0] , n))
          break
        else:
          cn = l[0]
          count += 1

  s = subsample
  edges_newtime = [((int(a[0]/s),a[1]) , (int(b[0]/s),b[1])) for a,b in edges]
  tbnew = nx.from_edgelist(edges_newtime , create_using=nx.DiGraph)
  newnodes_newtime = [(int(a[0]/s),a[1]) for a in newnodes]
  tbnew.add_nodes_from(newnodes_newtime)

  for n in tbnew.nodes:
    tbnew.nodes[n]['pt'] = tb.nodes[(int(n[0]*s),n[1])]['pt']
  return tbnew

"""
Given two TB's we want to compute TP/FP/FN for detections and edges across the whole timeseries.
Also, we should compute some stats about the length of valid tracking trajectories.
"""
def compare_tb(tb_gt,tb,scale=[1,1],dub=20):

  # if subsample is not None:
  #   tb_gt = subsample_graph(tb_gt,subsample=subsample)

  times = sorted(list(set([n[0] for n in tb_gt.nodes])))

  ## First match detections across all time
  ## `pts` are arrays of 3D coordinates. `track` maps from `pts` index to track label.
  matches = dict()
  for i in times:

    T = SimpleNamespace()
    # T.track0 = np.array([tb_gt.nodes[n]['track'] for n in tb_gt.nodes if n[0]==i])
    T.lab0   = np.array([n[1] for n in tb_gt.nodes if n[0]==i])
    pts0   = np.array([tb_gt.nodes[n]['pt'] for n in tb_gt.nodes if n[0]==i])
    # T.track1 = np.array([tb.nodes[n]['track'] for n in tb.nodes if n[0]==i])
    pts1   = np.array([tb.nodes[n]['pt'] for n in tb.nodes if n[0]==i])
    T.lab1   = np.array([n[1] for n in tb.nodes if n[0]==i])

    T.match = point_matcher.match_unambiguous_nearestNeib(pts0,pts1,scale=scale,dub=dub)
    matches[i] = T
    # print(f"T={i} P={T.match.precision:.5f} R={T.match.recall:.5f} F1={T.match.f1:.5f}")
    # ipdb.set_trace()


  ## Second, match on edges.
  ## Iterate over edges. first GT, then proposed.
  ## for each edge get parent -> match1 , child -> match2. assert (match1,match2) in proposed edges.
  edges_gt = np.zeros(len(tb_gt.edges))
  for n,e in enumerate(tb_gt.edges):
    t0 = e[0][0]
    t1 = e[1][0]
    ## get the index of e[0] and e[1]
    idx0 = np.argwhere(matches[t0].lab0==e[0][1])[0,0]
    idx1 = np.argwhere(matches[t1].lab0==e[1][1])[0,0]

    ## do the detection matches exist?
    if matches[t0].match.n_matched==0 and matches[t1].match.n_matched==0:
      edges_gt[n]=5
      continue
    elif matches[t0].match.n_matched==0:
      edges_gt[n]=3
      continue
    elif matches[t1].match.n_matched==0:
      edges_gt[n]=4
      continue

    matched_0 = matches[t0].match.gt_matched_mask[idx0]
    matched_1 = matches[t1].match.gt_matched_mask[idx1]

    ## if they both exist, then do those matches share an edge?
    if matched_0 and matched_1:
      l0 = matches[t0].lab1[matches[t0].match.gt2yp[idx0]]
      l1 = matches[t1].lab1[matches[t1].match.gt2yp[idx1]]
      e2 = ((t0,l0) , (t1,l1))
      matched_edge = tb.edges.get(e2 , None) ## None is default, but we are explicit

      ## woohoo! A match!
      if matched_edge is not None: ## must be explicit, because default value is empty set
        edges_gt[n]=1 ## maybe use (e,e2) for more info later

      ## out edge didn't match because of linking problem
      else:
        edges_gt[n]=2

    ## our edge didn't match because of mis-detection.
    elif matched_0:
      edges_gt[n]=3
    elif matched_1:
      edges_gt[n]=4
    else:
      edges_gt[n]=5 ## two mis-detections


  ## NOW do the same iteration again, but over the proposed (tb, lab1, pts1, etc)
  edges_prop = np.zeros(len(tb.edges))
  for n,e in enumerate(tb.edges):
    t0 = e[0][0]
    t1 = e[1][0]
    ## get the index of e[0] and e[1]
    idx0 = np.argwhere(matches[t0].lab1==e[0][1])[0,0]
    idx1 = np.argwhere(matches[t1].lab1==e[1][1])[0,0]
    ## do the detection matches exist?
    matched_0 = matches[t0].match.yp_matched_mask[idx0]
    matched_1 = matches[t1].match.yp_matched_mask[idx1]

    ## if they both exist, then do those matches share an edge?
    if matched_0 and matched_1:
      l0 = matches[t0].lab0[matches[t0].match.yp2gt[idx0]]
      l1 = matches[t1].lab0[matches[t1].match.yp2gt[idx1]]
      e2 = ((t0,l0) , (t1,l1))
      matched_edge = tb_gt.edges.get(e2 , None) ## None is default, but we are explicit

      ## woohoo! A match!
      if matched_edge is not None:
        edges_prop[n]=1 ## maybe use (e,e2) for more info later

      ## out edge didn't match because of linking problem
      else:
        edges_prop[n]=2

    ## our edge didn't match because of mis-detection.
    elif matched_0:
      edges_prop[n]=3
    elif matched_1:
      edges_prop[n]=4
    else:
      edges_prop[n]=5 ## two mis-detections


  assert (edges_prop==1).sum() == (edges_gt==1).sum()

  ## compute all the interesting whole-movie statistics about TP/FN/FP det's and edges
  S = SimpleNamespace() ## Scores

  ## detections

  S.n_matched_det  = sum([t.match.n_matched  for t in matches.values()])
  S.n_gt_det       = sum([t.match.n_gt       for t in matches.values()])
  S.n_proposed_det = sum([t.match.n_proposed for t in matches.values()]) ## n_proposed
  S.precision_det  = S.n_matched_det   / S.n_proposed_det
  S.recall_det     = S.n_matched_det   / S.n_gt_det
  S.f1_det         = S.n_matched_det*2 / (S.n_gt_det + S.n_proposed_det)

  ## edges

  S.n_matched_tra  = (edges_gt==1).sum() ## equiv to (edges_prop==1).sum()
  S.n_gt_tra       = edges_gt.shape[0]
  S.n_proposed_tra = edges_prop.shape[0]
  S.precision_tra  = S.n_matched_tra   / S.n_proposed_tra
  S.recall_tra     = S.n_matched_tra   / S.n_gt_tra
  S.f1_tra         = S.n_matched_tra*2 / (S.n_gt_tra + S.n_proposed_tra)

  # for k,v in S.__dict__.items():
  #   print(f"{k:12s}\t{v:.5f}")

  return matches , edges_gt , edges_prop , S
