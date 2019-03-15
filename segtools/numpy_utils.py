import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def broadcast_nonscalar_op(op, arr, subaxes, axes_full=None):
  "op idx->array must preserve shape of arr[idx]. less general than broadcast_nonscalar_func, but probs faster."
  
  arr = arr.copy()
  
  N = arr.ndim
  M = len(subaxes)
  if axes_full is None:
    axes_full = axes2str(range(N))
  subaxes = axes2str(subaxes)
  newaxes = move_axes_to_end(axes_full, subaxes)
  arr = perm(arr, axes_full, newaxes)

  for idx in np.ndindex(arr.shape[:N-M]):
    arr[idx] = op(idx)

  arr = perm(arr, newaxes, axes_full)
  return arr

def broadcast_nonscalar_func(func, arr, subaxes, axes_full=None):
  "func does not necessarily preserve ndim or shape. must return an array."
  N = arr.ndim
  M = len(subaxes)
  if axes_full is None:
    axes_full = axes2str(range(N))
  subaxes = axes2str(subaxes)
  newaxes = move_axes_to_end(axes_full, subaxes)
  arr = perm(arr, axes_full, newaxes)

  res = np.empty(arr.shape[:N-M],np.ndarray)
  for idx in np.ndindex(arr.shape[:N-M]):
    res[idx] = func(idx).tolist()
  res = np.array(res.tolist())

  res = perm(res, newaxes, axes_full)
  return res

def axes2str(axes):
  "idempotent."
  return ''.join([str(x) for x in axes])

def move_axes_to_end(allaxes, subaxes):
  s1 = allaxes.translate({ord(i):None for i in subaxes})
  s2 = s1 + subaxes
  return s2

def perm(arr,p1,p2):
  "permutation mapping p1 to p2 for use in numpy.transpose. elems must be unique."
  assert len(p1)==len(p2)
  perm = list(range(len(p1)))
  for i,p in enumerate(p2):
    perm[i] = p1.index(p)
  return arr.transpose(perm)

def perm2(arr,p1,p2):
  "permutation mapping p1 to p2 for use in numpy.transpose. elems must be unique."
  missing  = ''.join([p for p in p2 if p not in p1])
  newshape = arr.shape + tuple(1 for _ in missing)
  p1 = p1 + missing
  arr = arr.reshape(newshape)
  perm = list(range(len(p1)))
  for i,p in enumerate(p2):
    perm[i] = p1.index(p)
  return arr.transpose(perm)

def collapse(arr, axes=[[0,2],[1,3]]):
  sh = arr.shape
  perm = flatten(axes)
  arr = arr.transpose(perm)
  newshape = [np.prod([sh[i] for i in ax]) for ax in axes]
  arr = arr.reshape(newshape)
  return arr

def collapse2(arr, ax0, ax1):
  "e.g. collapse(arr, 'tczyx','ty,zx,c')"
  axes_list = [[ax0.index(x) for x in els] for els in ax1.split(',')]
  return collapse(arr, axes_list)

def merg(arr, ax=0):
  "given a list of axes, merge each one with it's successor."
  if type(ax) is list:
    assert all(ax[i] <= ax[i+1] for i in range(len(ax)-1))
    for i,axis in zip(range(100),ax):
      arr = merg(arr, axis-i)
  else: # int type  
    assert ax < arr.ndim-1
    sh = list(arr.shape)
    n = sh[ax]
    del sh[ax]
    sh[ax] *= n
    arr = arr.reshape(sh)
  return arr

def splt(arr, s1=10, ax=0):
  """
  split an array into more dimensions
  takes a list of split values and a list of axes and divides each axis into two new axes,
  where the first has a length given by the split value (which must by an even divisor of the original axis length)
  res = arange(200).reshape((2,100))
  res = splt(res, 5, 1)
  res.shape == (4,5,20)

  res = arange(3*5*7*11).reshape((3*5,7*11))
  res = splt(res, [3,7],[0,1])
  res.shape == (3,5,7,11)

  you can even list the same dimension multiple times
  res = arange(3*5*7*11).reshape((3*5*7,11))
  res = splt(res, [3,5],[0,0])
  res.shape == (3,5,7,11)
  """
  sh = list(arr.shape)
  if type(s1) is list and type(ax) is list:
    assert all(ax[i] <= ax[i+1] for i in range(len(ax)-1))
    for i,spl,axis in zip(range(100),s1, ax):
      arr = splt(arr, spl, axis+i)
  elif type(s1) is int and type(ax) is int:
    s2,r = divmod(sh[ax], s1)
    assert r == 0
    sh[ax] = s2
    sh.insert(ax, s1)
    arr = arr.reshape(sh)
  return arr

def multicat(lst):
  if type(lst[0]) is list:
    # type of list is list of list of...
    # apply recursively to each element. then apply to result.
    # apply recursively to every element except last
    res = [multicat(l) for l in lst[:-1]] + [lst[-1]]
    res = multicat(res)
  else:
    # lst is list of ndarrays. return an ndarray.
    res = np.concatenate(lst[:-1], axis=lst[-1])
  return res

def multistack(lst):
  if type(lst[0]) is list:
    # type of list is list of list of...
    # apply recursively to each element. then apply to result.
    # apply recursively to every element except last
    res = [multistack(l) for l in lst[:-1]] + [lst[-1]]
    res = multistack(res)
  # elif type(lst[0]) is int:
  #   # lst is list of ndarrays. return an ndarray.
  #   res = np.stack(lst[:-1], axis=lst[-1])
  else:
    res = np.stack(lst[:-1], axis=lst[-1])
  return res

def histogram(arr, bins=500):
  """
  We should extend the histogram function s.t. it always does this, and we don't have shitty plots. Also, if we give bins=-1, 
  we should just get back np.unique(arr, return_counts=True)...
  """
  arr = np.array(arr)
  if arr.ndim > 1:
      arr = arr.flatten()
  if bins==-1:
      vals, counts = np.unique(arr, return_counts=True)
      dx = 0
  else:
      hist_pimg = np.histogram(arr, bins=bins)
      counts, vals, dx = hist_pimg[0], (hist_pimg[1][1:] + hist_pimg[1][:-1])/2.0, hist_pimg[1][1]-hist_pimg[1][0]
      m = counts!=0
      counts = counts[m]
      vals   = vals[m]
  return counts, vals, dx

def sorted_uniques(img):
  a,b = np.unique(img, return_counts=True)
  counts = sorted(zip(a,b), key=lambda c: c[1])
  return counts

def argmax3d(img):
  "equivalent to divmod chaining"
  # alternative: return np.argwhere(img == img.max()) -- this returns all equiv maxima.
  return np.unravel_index(img.argmax(), img.shape)

def unique_across_axes(arr, axes):
  """
  keep `axes`. unique across axes not in `axes`
  """
  if type(axes) is int:
    axes = (axes,)
  sh = arr.shape
  sli = [slice(None, None),]*arr.ndim
  axes_shape = [sh[a] for a in axes]
  indices = np.indices(axes_shape)
  print(indices.shape)
  res = []
  for idx in indices.reshape((len(axes), -1)).T:
    for i in range(len(axes)):
      sli[axes[i]] = idx[i]
    res.append(np.unique(arr[sli]))
  res = np.array(res)
  res = res.reshape(axes_shape)
  return res


import numpy as np
from segtools.numpy_utils import perm2, splt, collapse2

def normalize(img,axs=(1,2,3),pc=[2,99.9],return_mima=False, clip=True):
  mi,ma = np.percentile(img,pc,axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  if clip:
    img = img.clip(0,1)
  if return_mima:
    return img,mi,ma
  else:
    return img

def normalize2(img,pmin=0,pmax=100,axs=None,return_mima=False,clip=True):
  if axs is None: axs = np.r_[0:img.ndim]
  mi,ma = np.percentile(img,[pmin, pmax],axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  if clip:
    img = img.clip(0,1)
  if return_mima:
    return img,mi,ma
  else:
    return img

def normalize3(img,pmin=0,pmax=100,axs=None,return_mima=False,clip=True):
  if axs is None: axs = np.r_[0:img.ndim]
  mi,ma = np.percentile(img,[pmin, pmax],axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  if clip: img = img.clip(0,1)
  if return_mima: return img,mi,ma
  else: return img

def pad_divisible(arr, dim, mult):
  s = arr.shape[dim]
  r = s % mult
  padding = np.zeros((arr.ndim,2), dtype=np.int)
  padding[dim,1] = (mult - r)%mult
  arr = np.pad(arr,padding,mode='constant')
  return arr

def pad_n_divisible(arr,dims,mults):
  for d,m in zip(dims,mults):
    arr = pad_divisible(arr,d,m)
  return arr

def pad_until(arr, dim, size):
  s = arr.shape[dim]
  r = size - s
  if r <= 0: return arr
  padding = np.zeros((arr.ndim,2), dtype=np.int)
  padding[dim,1] = r
  arr = np.pad(arr,padding,mode='constant')
  return arr

def plotgrid(lst, c=5):
  "each element of lst is a numpy array with axes 'SYX3' with last 3 channels RGB"
  res = np.stack(lst,0)
  res = pad_divisible(res, 1, c)
  r = res.shape[1]//c
  res = splt(res, r, 1)
  res = collapse2(res, 'iRCyxc','Ry,Cix,c')
  return res

