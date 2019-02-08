import sys
from copy import deepcopy
import itertools

import pandas as pd
import numpy as np

from scipy import ndimage as nd
from scipy.ndimage import label
from scipy.signal import gaussian
from sklearn.mixture import GaussianMixture
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure

from . import label_tools
from . import math_utils
from . import scores_dense
from . import color

## convert multiple nhls across time

def labs2nhls(labs, imgs, **kwargs):
  nhls = [hyp2nhl(labs[i], imgs[i], **kwargs) for i in range(labs.shape[0])]
  return nhls

## convert dense object representation into sparse representation (hyp 2 nhl)

def hyp2nhl(hyp, img=None, neighbordist=False, **kwargs):
  """
  TODO: allow local_center to choose between center of binary mask, and brightness-weighted center of hypothesis.
  TODO: check types on boundary to `moments_central` call.
  """
  if img is None:
    rps = measure.regionprops(hyp)
  else:
    rps = measure.regionprops(hyp, img)

  nhl = [regionprop2features(rp,hyp,img=img,**kwargs) for rp in rps]
  nhl  = sorted(nhl, key=lambda n: n['area'])

  if neighbordist:
    ## distribution of neighbor label values. works in 2d and 3d
    neibs = label_tools.dict_of_neighbor_distribution(hyp)
    for nuc in nhl:
      nuc['neighbor_distribution'] = neibs[nuc['label']]
      nuc['permimeter'] = sum(neibs[nuc['label']][1])

  return nhl

def regionprop2features(rp, hyp, img=None, time=None, moments=False):
  # centroid = rp.coords.mean(0)
  if hyp.ndim==3:
    minz,miny,minx,maxz,maxy,maxx = rp.bbox
    dims = [maxz-minz,maxy-miny,maxx-minx]
    # ss   = [slice(minz, maxz), slice(miny, maxy), slice(minx,maxx)]
  elif hyp.ndim==2:
    miny,minx,maxy,maxx = rp.bbox
    dims = [maxy-miny,maxx-minx]
    # ss   = [slice(miny, maxy), slice(minx,maxx)]

  features = {'label'    : rp.label,
              'area'     : int(rp.area),
              'centroid' : rp.centroid,
              'bbox'     : rp.bbox,
              'dims'     : dims,
              'slice'    : rp._slice,
              }

  if time is not None:
    features['time'] = time

  ## moments
  if moments and hyp.ndim==3:
    local_center = [coords[0]-minz, coords[1]-miny, coords[2]-minx]
    crop = hyp[minz:maxz, miny:maxy, minx:maxx].copy()
    crop = crop==rp.label
    mu = math_utils.moments_central(crop, map(int, local_center), 2)
    sig = math_utils.inertia_tensor(mu)
    eigvals, eigvecs = np.linalg.eig(sig)
    features['moments_central'] = mu
    features['inertia_tensor'] = sig
    features['inertia_tensor_eigvals'] = eigvals
    features['inertia_tensor_eigvecs'] = eigvecs
  elif moments and hyp.ndim==2:
    features['moments_central'] = rp.moments_central
    eigvals, eigvecs = np.linalg.eig(rp.inertia_tensor)
    features['inertia_tensor'] = rp.inertia_tensor
    features['inertia_tensor_eigvals'] = eigvals
    features['inertia_tensor_eigvecs'] = eigvecs
    rp.moments_central

  # if moments and img is not None and hyp.ndim==3:
  #   crop_img = img[minz:maxz, miny:maxy, minx:maxx].copy()
  #   crop_img[~crop] = 0
  #   mu2 = math_utils.moments_central(crop_img, map(int, local_center), 2)
  #   features['moments_img']   = mu2
  #   features['max_intensity'] = np.asscalar(rp.max_intensity)
  #   features['min_intensity'] = np.asscalar(rp.min_intensity)
  
  return features

## operate on nhl. convert sparse, structured representation into flat representation suitable for learning.

def nhl_mnmx(nhl, prop):
  areas = [n[prop] for n in nhl]
  a,b = max(areas), min(areas)
  return a,b

def nhl2dataframe(nhl, **kwargs):
  res = pd.DataFrame([flatten_nuc(x, **kwargs) for x in nhl])
  return res

def filter_nhls(nhls):
  def fil(n):
      if 3 < np.log2(n['area']):
          return True
      return False
  nhls2 = []
  for i,nhl in enumerate(nhls):
      nhl = [n for n in nhl if fil(n)]
      nhls2.append(nhl)
  return nhls2

def nhls2nucdict(nhls, f=lambda x: x):
  d = dict()
  for i, nhl in enumerate(nhls):
      for n in nhl:
          d[(i, n['label'])] = f(n)
  return d

## operate on single nuclei

def flatten_nuc(nuc, vecs=True, moments=True):
  ## operate on single nucle
  nhldict = deepcopy(nuc)
  
  for i in [0,1,2]:
    nhldict['dims{}'.format(i)] = nhldict['dims'][i]
  del nhldict['dims']

  for i in [0,1,2,3,4,5]:
    nhldict['bbox{}'.format(i)] = nhldict['bbox'][i]
  del nhldict['bbox']

  for i in [0,1,2]:
    nhldict['centroid{}'.format(i)] = nhldict['centroid'][i]
  del nhldict['centroid']

  for i in [0,1,2]:
    nhldict['eigvals_hyp{}'.format(i)] = nhldict['eigvals_hyp'][i]
  del nhldict['eigvals_hyp']

  if vecs:
    for i,j in itertools.product(*[[0,1,2],]*2):
      nhldict['eigvecs_hyp{}{}'.format(i,j)] = nhldict['eigvecs_hyp'][i,j]
    del nhldict['eigvecs_hyp']

  if moments:
    for i,j,k in itertools.product(*[[0,1,2],]*3):
      nhldict['moments_hyp{}{}{}'.format(i,j,k)] = nhldict['moments_hyp'][i,j,k]
    del nhldict['moments_hyp']

    for i,j,k in itertools.product(*[[0,1,2],]*3):
      nhldict['moments_img{}{}{}'.format(i,j,k)] = nhldict['moments_img'][i,j,k]
    del nhldict['moments_img']

  return nhldict

def nuc2slices_centroid(nuc, halfwidth=0, shift=0):
  centroid = [int(x) for x in nuc['centroid']]
  if not hasattr(halfwidth, '__len__'):
    halfwidth = [halfwidth]*len(centroid)
  if not hasattr(shift, '__len__'):
    shift = [shift]*len(centroid)

  def f(i):
    return slice(centroid[i] - halfwidth[i] + shift[i], centroid[i] + halfwidth[i] + shift[i])
  ss = [f(i) for i in range(len(centroid))]
  return ss

def nuc2slices(nuc, pad=0, shift=0):
  a,b,c,d,e,f = nuc['bbox']
  ss = slice(a-pad+shift,d+pad+shift), slice(b-pad+shift,e+pad+shift), slice(c-pad+shift,f+pad+shift)
  return ss

## operates on nuc mask

def fitgmm_mask(mask, **kwargs):
  nd = mask.ndim
  gm = GaussianMixture(**kwargs)
  ind = np.indices(mask.shape)
  ind_mask = ind[:, mask]
  ind_mask = ind_mask.reshape(nd, -1).T
  gm.fit(ind_mask)
  ind = ind.reshape(nd, -1).T
  preds = gm.predict_proba(ind)
  preds = preds.reshape(mask.shape + (-1,))
  return gm, preds

def fitgmm_to_nuc(nuc, pimg, hyp, pimgcut=0.5, n_components=2, spim=None):
  ss  = nuc2slices(nuc, 0)
  pimg = pimg[ss].copy()
  hyp = hyp[ss].copy()

  # plt.contour(ind[1,0], ind[2,0], preds[...,0])
  # plt.contour(ind[1,0], ind[2,0], preds[...,1])
  mask = hyp==nuc['label']
  gm, preds = fitgmm_mask(mask)
  hyp[mask] = 0
  # assert False
  for i in range(n_components):
    maski = (preds[...,i] > pimgcut) * mask
    hyp[maski] = i+1
    # hyp[maski] = hyp.max() + 1 ## why not this?
  if spim:
    img2 = pimg * m0
    img3 = pimg * m1
    spim.volshow([pimg, img2, img3, hyp], interpolation='nearest')
  # plt.contour(ind[]preds[...,1], alpha=0.5)
  return hyp

def grow_nuc(nuc, newmin, hyp, pimg):
  seed = hyp==nuc['label'] # & (img > 0.5) # isn't 2nd condition redundant?
  mask = pimg > newmin
  newlab = watershed(-pimg, seed, mask=mask)
  newlab *= nuc['label']
  return newlab

## Requires annotations

def fit_gmm_to_nhl(nhl, pimg, hyp, anno, cutoff=0.65):
  hyp = hyp.copy()
  for i in range(len(nhl)):
    nuc = nhl[i]
    n_components = anno[i]
    ss = nuc2slices(nuc, 0)
    hyp_crop  = hyp[ss].copy()
    mask = hyp_crop == nuc['label']
    hyp2 = fitgmm_to_nuc(nuc, pimg, hyp, pimgcut=cutoff, n_components=n_components)
    # print(np.unique(hyp2, return_counts=True))
    print(hyp.max(), hyp2[mask].max())
    hyp2[hyp2!=0] += hyp.max() # avoid label conflicts
    hyp[ss][mask] = hyp2[mask]
  # nhl2 = hyp2nhl(hyp, simple=True)
  # cut  = [n for n in nhl2 if n['area'] < 20]
  # hyp  = remove_nucs_hyp(cut, hyp)
  return hyp

def grow_all_nucs_in_nhl(nhl, anno, hyp, pimg):
  for i in range(len(nhl)):
    nuc = nhl[i]
    ann = anno[i]
    ss = nuc2slices(nuc, 30)
    hyp_ss = hyp[ss]
    pimg_ss = pimg[ss]
    newlab = grow_nuc(nuc, ann[1], hyp_ss, pimg_ss)
    mask = newlab!=0
    hyp[ss][mask] = newlab[mask]

## segmentation

def water_thresh_nuc(nuc, img, hyp, pimgcut=0.9, distcut=3.0):
  mask = hyp==nuc['label'] # & (img > 0.5) # isn't 2nd condition redundant?
  mask = mask.astype('int')
  distance = nd.distance_transform_edt(mask)
  mask = mask.astype('bool')
  c1 = pimgcut * img[mask].max()
  c2 = distcut
  mask2 = (img > c1) * mask
  if mask2.sum()==0:
    print("ERROR! ", img[mask].max())
  lab7 = nd.label(mask2 * (distance>=c2))[0]
  lab8 = watershed(-img, lab7, mask=mask)
  return lab8

def water_thresh_whole(nhl, pimg, hyp):
  hyp = hyp.copy()
  for nuc in nhl[-40:]:
    ss = nuc2slices(nuc, pad=4)
    hyp_crop  = hyp[ss].copy()
    pimg_crop = pimg[ss].copy()
    hyp2 = water_thresh_nuc(nuc, pimg_crop, hyp_crop, pimgcut=0.95, distcut=4.0)
    mask = hyp_crop == nuc['label']
    hyp[ss][mask] = hyp2[mask] + hyp.max()
  nhl2 = hyp2nhl(hyp, simple=True)
  cut  = [n for n in nhl2 if n['area'] < 20]
  hyp  = remove_nucs_hyp(cut, hyp)
  return hyp

def var_thresh(pimg, lab1, c2):
  nhl = hyp2nhl(lab1, img=pimg)
  def f(mi):
    if mi < c2:
      return 0.98 * mi
    else:
      return c2
  cmap = {n['label'] : f(n['max_intensity']) for n in nhl}
  cmap[0] = 1.0
  l1_max = color.recolor_from_mapping(lab1, cmap)
  m2 = pimg > l1_max
  return m2

def two_var_thresh(pimg, c1=0.5, c2=0.9):
  m1  = pimg>c1
  m2  = var_thresh(pimg, nd.label(m1)[0], c2)
  # m2  = pimg>0.9
  hyp = watershed(-pimg, nd.label(m2)[0], mask=m1)
  # hyp = nd.label(pimg>0.3)[0]
  nhl  = hyp2nhl(hyp, pimg)
  return nhl, hyp

## padding

@DeprecationWarning
def pad(img, w=10, mode='mean'):
  assert w > 0
  a,b,c = img.shape
  imgdbox = np.zeros((a+2*w, b+2*w, c+2*w))
  imgdbox[w:-w, w:-w, w:-w] = img
  imgdbox[:w] = img[0].mean()
  imgdbox[-w:] = img[-1].mean()
  imgdbox[:, :w] = img[:, 0].mean()
  imgdbox[:, -w:] = img[:, -1].mean()
  imgdbox[:, :, :w] = img[:, :, 0].mean()
  imgdbox[:, :, -w:] = img[:, :, -1].mean()
  return imgdbox

@DeprecationWarning
def pad4curation(img, hyp, nhl, pad=40):
  img_pad = pad_img(img, pad)
  hyp_pad = pad_img(hyp, pad)
  nhl_pad = pad_nhl(nhl, pad)
  return img_pad, hyp_pad, nhl_pad
  
@DeprecationWarning
def pad_nhl(nhl, pad=40):
  nhl_pad = deepcopy(nhl)
  q = pad
  for n in nhl_pad:
    a,b,c = n['centroid']
    n['centroid'] = [a+q, b+q, c+q]
    a,b,c,d,e,f = n['bbox']
    n['bbox'] = (a+q,b+q,c+q,d+q,e+q,f+q)
  return nhl_pad

@DeprecationWarning
def pad_img(img, pad=40, val=0):
  "pad with `val`"
  a,b,c = img.shape
  r = 2*pad
  q = pad
  img_pad = np.zeros((a+r, b+r, c+r), dtype=img.dtype) + val
  img_pad[q:q+a, q:q+b, q:q+c] = img
  return img_pad

## normalization

def normalize_percentile_to01(img, a, b):
  mn,mx = np.percentile(img, a), np.percentile(img, b)
  img = (1.0*img - mn)/(mx-mn)
  return img

## masking


## recoloring

@DeprecationWarning
def remove_nucs_hyp(nhl, hyp):
  hyp2 = hyp.copy()
  recolor = np.arange(0, hyp.max()+1, dtype=np.uint64)
  for n in nhl:
    recolor[n['label']] = 0
  hyp2 = recolor[hyp.flat].reshape(hyp.shape)
  return hyp2

@DeprecationWarning
def remove_easy_nuclei(hyp, biga):
  def f(i, c):
    if c=='1':
      return 0
    else:
      return i
  colormap = [f(i,c) for i,c in enumerate(biga)]
  colormap = np.array(colormap)
  arr = colormap[hyp.flat].reshape(hyp.shape)
  return arr
