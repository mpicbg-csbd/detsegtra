## anaconda
import numpy as np
import scipy.ndimage as nd
from scipy.ndimage.morphology import binary_dilation
# from scipy.ndimage.measurements import find_objects
from skimage.segmentation import find_boundaries
from . import nhl_tools
from numba import jit

@jit
def pixel_sharing_bipartite(lab1, lab2):
  assert lab1.shape == lab2.shape
  psg = np.zeros((lab1.max()+1, lab2.max()+1), dtype=np.int)
  for i in range(lab1.size):
    psg[lab1.flat[i], lab2.flat[i]] += 1
  return psg

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

@DeprecationWarning
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
