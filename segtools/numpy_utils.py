import numpy as np

def broadcast_nonscalar_op(op, arr, subaxes, axes_full=None):
  "op must preserve shape. less general than broadcast_nonscalar_func, but probs faster."
  
  arr = arr.copy()
  
  N = arr.ndim
  M = len(subaxes)
  if axes_full is None:
    axes_full = axes2str(range(N))
  subaxes = axes2str(subaxes)
  newaxes = move_axes_to_end(axes_full, subaxes)
  arr = perm(arr, axes_full, newaxes)

  for idx in np.ndindex(arr.shape[:N-M]):
    arr[idx] = op(arr[idx])

  arr = perm(arr, newaxes, axes_full)
  return arr

def broadcast_nonscalar_func(func, arr, subaxes, axes_full=None):
  "func must preserve ndim, but not necessarily shape."
  N = arr.ndim
  M = len(subaxes)
  if axes_full is None:
    axes_full = axes2str(range(N))
  subaxes = axes2str(subaxes)
  newaxes = move_axes_to_end(axes_full, subaxes)
  arr = perm(arr, axes_full, newaxes)

  res = np.empty(arr.shape[:N-M],np.ndarray)
  for idx in np.ndindex(arr.shape[:N-M]):
    res[idx] = func(arr[idx]).tolist()
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

def collapse(arr, axes=[[0,2],[1,3]]):
  sh = arr.shape
  perm = flatten(axes)
  arr = arr.transpose(perm)
  newshape = [np.prod([sh[i] for i in ax]) for ax in axes]
  arr = arr.reshape(newshape)
  return arr

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