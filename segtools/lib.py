import sys
from copy import deepcopy

import numpy as np
from numba import jit
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import gaussian
from scipy.ndimage import label
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from skimage import measure
import networkx as nx

import gputools
import spimagine

sys.path.insert(0, '/Users/colemanbroaddus/Desktop/Projects/cell_tracker/src/')
# import main.curation as curation
# import main.hypotheses as hypotheses
import main.drawing as drawing
# import main.graph_builder2 as graph_builder
import main.voronoi as voronoi

from . import segtools_simple
from . import color
from . import loc_utils as utils
from . import cell_view_lib as view

def rearrange_state(st):
  names = [
   'img_inp',
   'img_net',
   'img_pnet_inp',
   'img_pnet_inp_hyp',
   'img_pnet_net',
   'img_pnet_net_hyp',
   'img_pinp_inp',
   'img_pinp_inp_hyp',
   'img_pinp_net',
   'img_pinp_net_hyp',
   ]
  ordered_st = [st[n] for n in names]
  return ordered_st

def hyp2areahist(hyp):
  rps   = measure.regionprops(hyp)
  sizes = [int(rp.area) for rp in rps if rp.label != 0]
  res = utils.hist_dense(sizes, bins=40)
  return res

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
    minz,miny,minx,maxz,maxy,maxx = rp.bbox
    crop = hyp[minz:maxz, miny:maxy, minx:maxx].copy()
    crop = crop==rp.label
    # mu1 = moments_central(crop, [0,0,0], 1)
    # print(mu1)
    # sm = mu1[0,0,0]
    # local_center = [mu1[1,0,0]/sm, mu1[0,1,0]/sm, mu1[0,0,1]/sm]
    local_center = [coords[0]-minz, coords[1]-miny, coords[2]-minx]
    # local_centroid = rp.local_centroid
    mu = moments_central(crop, map(int, local_center), 2)
    sig = inertia_tensor(mu)
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
      mu2 = moments_central(crop_img, map(int, local_center), 2)
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
    mu = moments_central(crop, map(int, local_center), 2)
    sig = inertia_tensor(mu)
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
      mu2 = moments_central(crop_img, map(int, local_center), 2)
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

def hyp2coords(hyp):
  rps   = measure.regionprops(hyp)
  coords = [np.mean(rp.coords, axis=0) for rp in rps if rp.label != 0]
  coords = np.array(coords)
  return coords

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
  hyp  = label(pimg>th)[0]
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

def remove_nucs_hyp(nhl, hyp):
  hyp2 = hyp.copy()
  recolor = np.arange(0, hyp.max()+1, dtype=np.uint64)
  for n in nhl:
    recolor[n['label']] = 0
  hyp2 = recolor[hyp.flat].reshape(hyp.shape)
  return hyp2

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

def remove_easy_nuclei(hyp, biga):
  def f(i, c):
    if c=='1':
      return 0
    else:
      return i
  colormap = [f(i,c) for i,c in enumerate(biga)]
  colormap = np.array(colormap)
  arr = drawing.highlight_replace(hyp, colormap)
  return arr

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

def moments_simple_2nd(img):
  mu = np.zeros((3,)*3)
  mu[0,0,0] = np.sum(img)
  ind = np.indices(img.shape)
  mu[1,0,0] = np.sum(ind[0]*img)
  mu[0,1,0] = np.sum(ind[1]*img)
  mu[0,0,1] = np.sum(ind[2]*img)
  mu[1,1,0] = np.sum(ind[0]*ind[1]*img)
  mu[0,1,1] = np.sum(ind[1]*ind[2]*img)
  mu[1,0,1] = np.sum(ind[0]*ind[2]*img)
  mu[1,1,1] = np.sum(ind[0]*ind[1]*ind[2]*img)
  return mu

@jit('f4[:,:,:](f4[:,:,:],f4[:],u1)')
def moments_central(image, cen, order):
  # cdef Py_ssize_t p, q, r, c
  mu = np.zeros((order + 1, order + 1, order + 1), dtype=np.double)
  # cdef double val, dr, dc, dcp, drq
  cx1, cx2, cx3 = cen
  for x1 in range(image.shape[0]):
      x1 = x1 - cx1
      for x2 in range(image.shape[1]):
          x2 = x2 - cx2
          for x3 in range(image.shape[2]):
            x3 = x3 - cx3
            
            val = image[x1, x2, x3]
            
            dcx1 = 1
            for p1 in range(order + 1):
              dcx2 = 1
              for p2 in range(order + 1):
                  dcx3 = 1
                  for p3 in range(order + 1):
                      mu[p1, p2, p3] += val * dcx1 * dcx2 * dcx3
                      dcx3 *= x3
                  dcx2 *= x2
              dcx1 *= x1
  return np.asarray(mu)

def inertia_tensor(mu):
  """
  mu = moments_central
  """
  inertia_tensor = [[mu[2,0,0], mu[1,1,0], mu[1,0,1]],
                    [mu[1,1,0], mu[0,2,0], mu[0,1,1]],
                    [mu[1,0,1], mu[0,1,1], mu[0,0,2]]]
  inertia_tensor = np.array(inertia_tensor)
  inertia_tensor /= mu[0,0,0]
  return inertia_tensor

def ellipse(n=100, z=[1,1,1]):
  cutoff = 3*n
  img = np.zeros((n,n,n), np.float)
  elli = np.indices(img.shape, np.float)
  elli -= np.array([(n-1)/2,(n-1)/2,(n-1)/2]).reshape((3,1,1,1))
  elli *= np.array(z).reshape((3,1,1,1))
  elli = np.sum(elli**2, axis=0)
  mask = (elli < cutoff).astype(np.float)
  return mask, elli

def get_cube_from_transform(img, tcube):
  cube = drawing.get_cube_from_transform(img, tcube).copy()
  # slt = drawing.get_slices_from_transform(img, tcube)
  # return slt
  return cube

def highlight_img_by_feature(hyp, nhl, zeroval=1):
  """
  color nuclei by an arbitrary function of their features
  nhl: list of nuclei (dictionaries) which have 'label' and 'color' keys
  """
  cmap_arr = np.ones(int(hyp.max() + 1))
  for n in nhl:
    l = n['label']
    cmap_arr[l] = n['color']
  cmap_arr[0] = zeroval

  if cmap_arr.ndim==1:
    highlighted = drawing.highlight_replace(hyp, cmap_arr)
  elif cmap_arr.ndim==2:
    highlighted = drawing.highlight_replace(hyp, cmap_arr)
  else:
    raise IndexError

  return highlighted

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

def lap_of_gaus_nd(x, sig=6):
  "x is a vector in Real Euclidean space."
  e = np.e
  π = np.pi
  σ = sig

  r2 = (x*x).sum()/(2*σ**2)
  m1 = 1 / (π * σ**4)
  m2 = 1 - r2
  m3 = e**(-r2)
  m4 = m1*m2*m3
  return m4

def conv_log_3d(img, sig=2, w=10):
  dom = np.indices((w,)*3)
  dom = dom - (w-1)/2
  res = [lap_of_gaus_nd(x, sig) for x in dom.reshape(3,-1).T]
  res = np.array(res).reshape(dom.shape[1:])
  res = res/res.sum()
  # img = img.astype(np.float32)
  img2 = gputools.convolve(img, res)
  return img2

def argmax3d(img):
  "equivalent to divmod chaining"
  return np.unravel_index(img.argmax(), img.shape)

def patchify(img, patch_shape):
  """
  From StackOverflow https://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image?noredirect=1&lq=1
  eg:
  out = patchify(x, (S,S)).max(axis=(3,4))
  """
  a, X, Y, b = img.shape
  x, y = patch_shape
  shape = (a, X - x + 1, Y - y + 1, x, y, b)
  a_str, X_str, Y_str, b_str = img.strides
  strides = (a_str, X_str, Y_str, X_str, Y_str, b_str)
  return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

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

def mask2cmask(mask, dil_iter=2, sig=3.0):
  mask = binary_dilation(mask, iterations=2).astype('float')
  hx = gaussian(9, 3.0)
  mask = gputools.convolve_sep3(mask, hx, hx, hx)
  mask = mask/mask.max()
  mask = 1-mask
  return mask

def get_slices_from_transform(img, tcube):
  """
  specify a spimagine.TransformData and full-sized image
  get the image cube inside the bounding box.
  """
  zhw,yhw,xhw = np.array(img.shape)/2
  tcube.bounds
  xmin = int((1 + tcube.bounds[0])*xhw)
  xmax = int((1 + tcube.bounds[1])*xhw)
  ymin = int((1 + tcube.bounds[2])*yhw)
  ymax = int((1 + tcube.bounds[3])*yhw)
  zmin = int((1 + tcube.bounds[4])*zhw)
  zmax = int((1 + tcube.bounds[5])*zhw)
  slt = slice(zmin,zmax), slice(ymin,ymax), slice(xmin,xmax)
  return slt

def transform2slices(trans, shape):
  """
  specify a spimagine.TransformData and full-sized image
  get the image cube inside the bounding box.
  """
  zhw,yhw,xhw = np.array(shape)/2
  trans.bounds
  xmin = int((1 + trans.bounds[0])*xhw)
  xmax = int((1 + trans.bounds[1])*xhw)
  ymin = int((1 + trans.bounds[2])*yhw)
  ymax = int((1 + trans.bounds[3])*yhw)
  zmin = int((1 + trans.bounds[4])*zhw)
  zmax = int((1 + trans.bounds[5])*zhw)
  slt = slice(zmin,zmax), slice(ymin,ymax), slice(xmin,xmax)
  return slt

def nhl_mnmx(nhl, prop):
  areas = [n[prop] for n in nhl]
  a,b = max(areas), min(areas)
  return a,b

## voronoi method -- way too slow. 4 mins / img.
def voronoi(pimg, nhl):
  coords = np.array([n['coords'] for n in nhl])
  vor    = voronoi.voronoi_kd(coords, img.shape, maxdist=20)
  mask   = pimg > 0.5
  lab    = vor[0]*mask
  nhl2   = hyp2nhl(lab, simple=True)
  return nhl2, lab

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
  lab, ncells = label(img_crop > newthresh)
  spimagine.volshow([img_crop, hyp_crop, lab], interpolation='nearest')
  input('quit?')

def nuc_grid_plot(img, nhl):
  if len(nhl) > 100:
    print("Too many nuclei! Try again w len(nhl) < 100.")
    return False
  def f(i):
    img_crop = nuc2img(nhl[i], img, 4)
    lab, ncells = label(img_crop > 0.92)
    lab = lab.sum(2)
    return lab
  patches = [f(i) for i in range(len(nhl))]
  
  coords = np.indices((4,5))*30
  plotimg = patchmaker.piece_together_ragged_2d(patches, coords.reshape(2,-1).T)
  plt.imshow(plotimg)

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

def water_thresh_nuc(nuc, img, hyp, pimgcut=0.9, distcut=3.0):
  mask = hyp==nuc['label'] # & (img > 0.5) # isn't 2nd condition redundant?
  mask = mask.astype('int')
  distance = ndi.distance_transform_edt(mask)
  mask = mask.astype('bool')
  c1 = pimgcut * img[mask].max()
  c2 = distcut
  mask2 = (img > c1) * mask
  if mask2.sum()==0:
    print("ERROR! ", img[mask].max())
  lab7 = label(mask2 * (distance>=c2))[0]
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

@DeprecationWarning
def remove_border_cells(nhl, hyp):
  assert False
  s1 = (0,0,0) + hyp.shape
  s1 = np.array(s1)
  s2 = np.array([n['bbox'] for n in nhl])
  badinds = (s1==s2).any(1)
  remove_these = np.array(nhl)[badinds]
  hyp = remove_nucs_hyp(remove_these, hyp)
  nhl = hyp2nhl(hyp, simple=True)
  return nhl, hyp

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

def curate_nhl(w, nhl, img, hyp, pp):
  def crops(i):
    nuc = nhl[i]
    ss = nuc2slices_centroid(nuc, pp//2)
    img_crop = img[ss].copy()
    hyp_crop = hyp[ss].copy()
    mask = hyp_crop==nuc['label']
    img_crop[mask] *= 2.0
    return img_crop, hyp_crop

  def nextnuc(i):
    imgc, hypc = crops(i)
    update_spim(w, 0, imgc)
    # w.glWidget.renderer.update_data(imgc)
    # w.transform.setPos(1)
    # w.glWidget.renderer.update_data(hypc)
    # w.transform.setPos(0)
    # w.glWidget.refresh()
    ans = input("How many nuclei do you see? :: ")
    ans = (ans, w.transform.maxVal)
    return ans

  biganno = ['no idea' for _ in nhl]

  imgc, hypc = crops(0)
  w.transform.setMax(imgc.max())
  ans = nextnuc(0)
  if ans == 'q':
    return None
  else:
    biganno[0] = ans
  i = 1
  while i < len(nhl):
    ans = nextnuc(i)
    if ans == 'q':
        break
    elif ans == 'k':
        print("Undo...")
        i -= 1
    else:
        biganno[i] = ans
        i += 1
  return biganno

def grow_nuc(nuc, newmin, hyp, pimg):
  seed = hyp==nuc['label'] # & (img > 0.5) # isn't 2nd condition redundant?
  mask = pimg > newmin
  newlab = watershed(-pimg, seed, mask=mask)
  newlab *= nuc['label']
  return newlab

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
  m2  = var_thresh(pimg, label(m1)[0], c2)
  # m2  = pimg>0.9
  hyp = watershed(-pimg, label(m2)[0], mask=m1)
  # hyp = label(pimg>0.3)[0]
  nhl  = hyp2nhl(hyp, pimg)
  return nhl, hyp

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

def sub_block_apply(func, img, sub_blocks=(1,1,1)):
  a,b,c = img.shape
  n1, n2, n3 = sub_blocks
  ar = list(range(0, a+1, a//n1)); ar[-1]=None
  br = list(range(0, b+1, b//n2)); br[-1]=None
  cr = list(range(0, c+1, c//n3)); cr[-1]=None
  res = np.zeros_like(img)
  for i in range(n1):
    for j in range(n2):
      for k in range(n3):
        ss = (slice(ar[i], ar[i+1]), slice(br[j], br[j+1]), slice(cr[k], cr[k+1]))
        res[ss] = func(img[ss])
  return res

def normalize_percentile_to01(img, a, b):
  mn,mx = np.percentile(img, a), np.percentile(img, b)
  img = (1.0*img - mn)/(mx-mn)
  return img

def update_spim(w, i, cube):
  w.glWidget.dataModel[i][...] = cube
  w.glWidget.dataPosChanged(i)
