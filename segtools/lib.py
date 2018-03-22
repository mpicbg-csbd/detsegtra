import sys
from copy import deepcopy

import numpy as np

from scipy import ndimage as nd

from scipy.signal import gaussian

from sklearn.mixture import GaussianMixture

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure

import gputools
import spimagine

from . import voronoi
from . import math_utils
from . import segtools_simple
from . import color
from . import loc_utils
from . import cell_view_lib as view


@DeprecationWarning
def pixel_confusion(pimg, labelimg, threshold):
  """
  data specific! labeling has bg=1, nuclei=2, unknown=0. pimg only has one (nuclei) channel!
  labeling: is sparse labeling image with same shape as pimg.
  pimg: is a probability map for nuclei pixels
  """
  assert np.unique(labelimg).tolist()==[0,1,2]
  assert pimg.min()>=0 and pimg.max()<=1
  nuclei_mask = (pimg > threshold).astype('uint8') + 1
  m = labelimg==1
  c11 = (nuclei_mask[m]==1).sum() # true negative
  c12 = (nuclei_mask[m]==2).sum() # false positive
  m = labelimg==2
  c21 = (nuclei_mask[m]==1).sum() # false negative
  c22 = (nuclei_mask[m]==2).sum() # true positive
  confusion = [c11, c12, c21, c22]
  return confusion

## operate on hyp

def hyp2areahist(hyp):
  rps   = measure.regionprops(hyp)
  sizes = [int(rp.area) for rp in rps if rp.label != 0]
  res = loc_utils.hist_dense(sizes, bins=40)
  return res

def hyp2coords(hyp):
  rps   = measure.regionprops(hyp)
  coords = [np.mean(rp.coords, axis=0) for rp in rps if rp.label != 0]
  coords = np.array(coords)
  return coords

def hyp2nhl_2d(hyp, img=None, time=None, simple=False):
  """
  TODO: allow local_center to choose between center of binary mask, and brightness-weighted center of hypothesis.
  TODO: check types on boundary to `moments_central` call.
  """
  if img is None:
    rps = measure.regionprops(hyp)
  else:
    rps = measure.regionprops(hyp, img)

  def f(rp):
    coords = np.mean(rp.coords, axis=0).tolist()
    miny,minx,maxy,maxx = rp.bbox
    crop = hyp[miny:maxy, minx:maxx].copy()
    crop = crop==rp.label
    # mu1 = moments_central(crop, [0,0,0], 1)
    # print(mu1)
    # sm = mu1[0,0,0]
    # local_center = [mu1[1,0,0]/sm, mu1[0,1,0]/sm, mu1[0,0,1]/sm]
    local_center = [coords[0]-miny, coords[1]-minx]
    # local_centroid = rp.local_centroid
    mu = rp.moments_central #math_utils.moments_central(crop, map(int, local_center), 2)
    sig = rp.inertia_tensor #math_utils.inertia_tensor(mu)
    eigvals, eigvecs = np.linalg.eig(sig)
    features = {'label'   : rp.label,
                'area'    : int(rp.area),
                'coords'  : coords,
                'bbox'    : rp.bbox,}

    if time is not None:
      features['time'] = time

    if simple:
      return features
    
    # assert False

    extra = {'moments_hyp' : mu,
             'eigvals_hyp' : eigvals,
             'eigvecs_hyp' : eigvecs, }

    features = {**features, **extra}

    if img is not None:
      crop_img = img[miny:maxy, minx:maxx].copy()
      crop_img[~crop] = 0
      # mu2 = math_utils.moments_central(crop_img, map(int, local_center), 2)
      mu2 = rp.moments_central
      # print(mu2, type(mu2))
      features['moments_img']   = mu2
      # print(rp.max_intensity, type(rp.max_intensity))
      # mn, mx = np.asscalar(rp.max_intensity), np.asscalar(rp.min_intensity)
      features['max_intensity'] = np.asscalar(rp.max_intensity)
      features['min_intensity'] = np.asscalar(rp.min_intensity)
    
    return features

  nhl = [f(rp) for rp in rps]
  nhl  = sorted(nhl, key=lambda n: n['area'])
  return nhl

def hyp2nhl(hyp, img=None, time=None, simple=False):
  """
  TODO: allow local_center to choose between center of binary mask, and brightness-weighted center of hypothesis.
  TODO: check types on boundary to `moments_central` call.
  """
  if img is None:
    rps = measure.regionprops(hyp)
  else:
    rps = measure.regionprops(hyp, img)

  def f(rp):
    coords = np.mean(rp.coords, axis=0).tolist()
    minz,miny,minx,maxz,maxy,maxx = rp.bbox
    crop = hyp[minz:maxz, miny:maxy, minx:maxx].copy()
    crop = crop==rp.label
    # mu1 = moments_central(crop, [0,0,0], 1)
    # print(mu1)
    # sm = mu1[0,0,0]
    # local_center = [mu1[1,0,0]/sm, mu1[0,1,0]/sm, mu1[0,0,1]/sm]
    local_center = [coords[0]-minz, coords[1]-miny, coords[2]-minx]
    # local_centroid = rp.local_centroid
    mu = math_utils.moments_central(crop, map(int, local_center), 2)
    sig = math_utils.inertia_tensor(mu)
    eigvals, eigvecs = np.linalg.eig(sig)
    features = {'label'   : rp.label,
                'area'    : int(rp.area),
                'coords'  : coords,
                'bbox'    : rp.bbox,}

    if time is not None:
      features['time'] = time

    if simple:
      return features
    
    extra = {'moments_hyp' : mu,
             'eigvals_hyp' : eigvals,
             'eigvecs_hyp' : eigvecs, }

    features = {**features, **extra}

    if img is not None:
      crop_img = img[minz:maxz, miny:maxy, minx:maxx].copy()
      crop_img[~crop] = 0
      mu2 = math_utils.moments_central(crop_img, map(int, local_center), 2)
      # print(mu2, type(mu2))
      features['moments_img']   = mu2
      # print(rp.max_intensity, type(rp.max_intensity))
      # mn, mx = np.asscalar(rp.max_intensity), np.asscalar(rp.min_intensity)
      features['max_intensity'] = np.asscalar(rp.max_intensity)
      features['min_intensity'] = np.asscalar(rp.min_intensity)
    
    return features

  nhl = [f(rp) for rp in rps]
  nhl  = sorted(nhl, key=lambda n: n['area'])
  return nhl

## operate on nhl

def nhl_mnmx(nhl, prop):
  areas = [n[prop] for n in nhl]
  a,b = max(areas), min(areas)
  return a,b

## operate on single nuclei

def nuc2X(nuc):
  """
  Useful when you want to train a classifier on nuclei to predict segmentation type.
  """
  ar = nuc['area']
  c0 = nuc['coords'][0]
  c1 = nuc['coords'][1]
  c2 = nuc['coords'][2]
  b0 = nuc['bbox'][0]
  b1 = nuc['bbox'][1]
  b2 = nuc['bbox'][2]
  # mh000 = nuc['moments_hyp'][0,0,0]
  # mh100 = nuc['moments_hyp'][1,0,0]
  # mh010 = nuc['moments_hyp'][0,1,0]
  # mh001 = nuc['moments_hyp'][0,0,1]
  # mh100 = nuc['moments_hyp'][1,0,0]
  # mh010 = nuc['moments_hyp'][0,1,0]
  # mh001 = nuc['moments_hyp'][0,0,1]
  e0 = nuc['eigvals_hyp'][0]
  e1 = nuc['eigvals_hyp'][1]
  e2 = nuc['eigvals_hyp'][2]
  mx = nuc['max_intensity'][0]
  mn = nuc['min_intensity'][0]
  ti = nuc.get('time', 0)
  return [ar, c0, c1, c2, b0, b1, b2, e0, e1, e2, mx, mn, ti]

def nuc2slices_centroid(nuc, halfwidth):
  a,b,c = map(int, nuc['coords'])
  hw=halfwidth
  ss = (slice(a-hw, a+hw), slice(b-hw, b+hw), slice(c-hw, c+hw))
  return ss

def nuc2slices(nuc, pad):
  a,b,c,d,e,f = nuc['bbox']
  return slice(a-pad,d+pad), slice(b-pad,e+pad), slice(c-pad,f+pad)

def nuc2img(nuc, img, pad):
  ss = nuc2slices(nuc, pad=pad)
  return img[ss].copy()

def rethresh_nuc(nuc, img, hyp, pad, newthresh=0.75):
  img_crop = nuc2img(nuc, img, pad)
  hyp_crop = nuc2img(nuc, hyp, pad)
  lab, ncells = nd.label(img_crop > newthresh)
  spimagine.volshow([img_crop, hyp_crop, lab], interpolation='nearest')
  input('quit?')

def fitgmm(nuc, pimg, hyp, pimgcut=0.5, n_components=2, show=False):
  gm = GaussianMixture(n_components=n_components)
  ss  = nuc2slices(nuc, 0)
  pimg = pimg[ss].copy()
  hyp = hyp[ss].copy()
  ind = np.indices(hyp.shape)
  mask = hyp==nuc['label']
  ind = ind[:, mask]
  ind = ind.reshape(3, -1).T
  gm.fit(ind)
  ind = np.indices(hyp.shape)
  ind = ind.reshape(3, -1).T
  preds = gm.predict_proba(ind)
  preds = preds.reshape(hyp.shape + (n_components,))
  # preds = preds.sum(0)
  ind = ind.reshape((3,) + hyp.shape)
  # plt.contour(ind[1,0], ind[2,0], preds[...,0])
  # plt.contour(ind[1,0], ind[2,0], preds[...,1])
  hyp[mask] = 0
  # assert False
  for i in range(n_components):
    maski = (preds[...,i] > pimgcut) * mask
    hyp[maski] = i+1
  # w2.glWidget.renderer.set_data(np.array([pimg, img2,img3]))
  if show:
    img2 = pimg * m0
    img3 = pimg * m1
    spimagine.volshow([pimg, img2, img3, hyp], interpolation='nearest')
  # plt.contour(ind[]preds[...,1], alpha=0.5)
  return hyp

def grow_nuc(nuc, newmin, hyp, pimg):
  seed = hyp==nuc['label'] # & (img > 0.5) # isn't 2nd condition redundant?
  mask = pimg > newmin
  newlab = watershed(-pimg, seed, mask=mask)
  newlab *= nuc['label']
  return newlab

## requires anno

def anno2y(anno):
  anno2 = anno.copy()
  anno_vals = ['0', '1', '1.5', '2', 'h']
  for i,v in enumerate(anno_vals):
    anno2[anno == v]=i
  anno2 = np.array(anno2, dtype=np.float)
  return anno2

  # mask = np.zeros_like(stack['hyp'], dtype=np.bool)
  # for n in nhl_area[-50:]:
  #     mask += stack['hyp']==n['label']
  # pimgcopy = stack['pimg'].copy()
  # pimgcopy[mask] *= 1.5
  # w = spimagine.volshow(pimgcopy)
  # view.moveit(w)

def run_gmm(nhl, pimg, hyp, anno, cutoff=0.65):
  hyp = hyp.copy()
  for i in range(len(nhl)):
    nuc = nhl[i]
    n_components = anno[i]
    ss = nuc2slices(nuc, 0)
    hyp_crop  = hyp[ss].copy()
    mask = hyp_crop == nuc['label']
    hyp2 = fitgmm(nuc, pimg, hyp, pimgcut=cutoff, n_components=n_components)
    # print(np.unique(hyp2, return_counts=True))
    print(hyp.max(), hyp2[mask].max())
    hyp2[hyp2!=0] += hyp.max() # avoid label conflicts
    hyp[ss][mask] = hyp2[mask]
  # nhl2 = hyp2nhl(hyp, simple=True)
  # cut  = [n for n in nhl2 if n['area'] < 20]
  # hyp  = remove_nucs_hyp(cut, hyp)
  return hyp

def grow_all(nhl, anno, hyp, pimg):
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

def var_thresh(pimg, lab1, t2):
  nhl = hyp2nhl(lab1, img=pimg)
  def f(mi):
    if mi < t2:
      return 0.98 * mi
    else:
      return t2
  cmap = {n['label'] : f(n['max_intensity']) for n in nhl}
  cmap[0] = 1.0
  l1_max = color.apply_mapping(lab1, cmap)
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

def spectral_clustering_seg(img, n_clusters=8, threshold=150):
    """
    uses spectral clustering on pixel graph to split connected components.
    only included here for posterity
    shitty method for nuclei. also very slow. img is uint16? HypothesisImage.
    """
    from sklearn.feature_extraction.image import img_to_graph
    from sklearn.cluster import spectral_clustering
    bimg = gputools.blur(img)
    plt.imshow(bimg)
    mask=bimg>threshold
    graph = img_to_graph(bimg, mask=mask)
    graph.data = np.exp(-graph.data/graph.data.std())
    labels = spectral_clustering(graph, n_clusters=8)
    labelsim = np.zeros(bimg.shape)
    labelsim[mask] = labels
    return labelsim

@DeprecationWarning
def pimg2hyp(pimg, th, bl=1.0, minsize=34, maxsize=None, minfilter=0):
  """
  segment your probability maps. normalizes to [0,1] just before thresholding.
  """
  # TODO: Can we compose the gpu ops to avoid moving data twice?
  pimg = pimg.copy()

  if minfilter > 0:
    pimg = gputools.min_filter(pimg, size=minfilter)
    
  if bl > 0:
    hx = gaussian(9, bl)
    pimg = gputools.convolve_sep3(pimg, hx, hx, hx)

  pimg = pimg/pimg.max()
  hyp  = nd.label(pimg>th)[0]
  nhl  = hyp2nhl(hyp, simple=True)
  def f(n):
    a = n['area']
    if minsize > a:
      return True
    if maxsize and maxsize < a:
      return True
    return False
  cut = filter(f, nhl)
  hyp = remove_nucs_hyp(cut, hyp)
  return hyp

## padding

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

def pad4curation(img, hyp, nhl, pad=40):
  img_pad = pad_img(img, pad)
  hyp_pad = pad_img(hyp, pad)
  nhl_pad = pad_nhl(nhl, pad)
  return img_pad, hyp_pad, nhl_pad
  
def pad_nhl(nhl, pad=40):
  nhl_pad = deepcopy(nhl)
  q = pad
  for n in nhl_pad:
    a,b,c = n['coords']
    n['coords'] = [a+q, b+q, c+q]
    a,b,c,d,e,f = n['bbox']
    n['bbox'] = (a+q,b+q,c+q,d+q,e+q,f+q)
  return nhl_pad

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

def mask_nhl(nhl, hyp):
  labels = [n['label'] for n in nhl]
  mask = mask_labels(labels, hyp)
  return mask

def mask_labels(labels, hyp):
  mask = hyp.copy()
  recolor = np.zeros(hyp.max()+1, dtype=np.bool)
  for l in labels:
    recolor[l] = True
  mask = recolor[hyp.flat].reshape(hyp.shape)
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

def mask2cmask(mask, dil_iter=2, sig=3.0):
  mask = nd.binary_dilation(mask, iterations=2).astype('float')
  hx = gaussian(9, 3.0)
  mask = gputools.convolve_sep3(mask, hx, hx, hx)
  mask = mask/mask.max()
  mask = 1-mask
  return mask


## recoloring

def remove_nucs_hyp(nhl, hyp):
  hyp2 = hyp.copy()
  recolor = np.arange(0, hyp.max()+1, dtype=np.uint64)
  for n in nhl:
    recolor[n['label']] = 0
  hyp2 = recolor[hyp.flat].reshape(hyp.shape)
  return hyp2

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
