## anaconda
import numpy as np
# import scipy.spatial as spatial
import scipy.ndimage as nd
# from skimage import measure
# from numba import jit
from scipy.ndimage.morphology import binary_dilation
# from scipy.ndimage.measurements import find_objects

## manual install
# from pykdtree.kdtree import KDTree as pyKDTree

## my lib
# from .math_utils import xyz2rthetaphi
from .scores_dense import pixel_sharing_bipartite
from . import nhl_tools


## borders / boundaries

def pixelgraph_edge_distribution_unique(lab):
    "must be dense."
    nhl = nhl_tools.hyp2nhl(lab)
    dist = dict()
    lab2 = np.pad(lab,4,mode='constant')
    for nuc in nhl:
        ss  = nhl_tools.nuc2slices(nuc, pad=4, shift=4)
        l   = nuc['label']
        m   = lab2[ss]==l
        m2  = binary_dilation(m) # by default uses distance<=1 neighborhood.
        vals, cnts = np.unique(lab2[ss][m2 & ~m], return_counts=True)
        dist[l] = vals.tolist(), cnts.tolist()
    return dist

def pixelgraph_edge_distribution(lab, neibdist=[1,1,1]):

    hist = np.zeros((lab.max()+1, lab.max()+1), dtype=np.int64)
    if len(neibdist)==2:
        nd0,nd1 = neibdist
    elif len(neibdist)==3:
        nd0,nd1,nd2 = neibdist

    psg = pixel_sharing_bipartite(lab[nd0:, :], lab[:-nd0, :])
    a,b = psg.shape
    hist[:a,:b] += psg

    psg = pixel_sharing_bipartite(lab[:, nd1:], lab[:, :-nd1])
    a,b = psg.shape
    hist[:a,:b] += psg

    if lab.ndim == 3:
        psg = pixel_sharing_bipartite(lab[:, :, nd2:], lab[:, :, :-nd2])
        a,b = psg.shape
        hist[:a, :b] += psg

    return psg

def boundary_image(lab):
    "max value = 2*lab.ndim. works in 2d and 3d."
    res = np.zeros(lab.shape)
    m1  = lab[1:, :] == lab[:-1, :]
    m2  = lab[:, 1:] == lab[:, :-1]
    res[:-1] = m1
    res[1:]  += m1
    res[:,1:] += m2
    res[:,:-1] += m2
    if lab.ndim==3:
        m3  = lab[:, :, 1:] == lab[:, :, :-1]
        res[:,:,1:] += m3
        res[:,:,:-1] += m3
    return res

## masking

def mask_nhl(nhl, hyp):
  labels = [n['label'] for n in nhl]
  mask = mask_labels(labels, hyp)
  return mask

def mask_labels(labels, lab):
  mask = lab.copy()
  recolor = np.zeros(lab.max()+1, dtype=np.bool)
  for l in labels:
    recolor[l] = True
  mask = recolor[lab.flat].reshape(lab.shape)
  return mask

def mask_border_objs(hyp, bg_id=0):
  id_set = set()
  id_set = id_set | set(np.unique(hyp[0])) - {bg_id}
  id_set = id_set | set(np.unique(hyp[-1])) - {bg_id}
  id_set = id_set | set(np.unique(hyp[:,0])) - {bg_id}
  id_set = id_set | set(np.unique(hyp[:,-1])) - {bg_id}
  if hyp.ndim == 3:
    id_set = id_set | set(np.unique(hyp[:,:,0])) - {bg_id}
    id_set = id_set | set(np.unique(hyp[:,:,-1])) - {bg_id}
  mask = mask_labels(id_set, hyp)
  return mask
