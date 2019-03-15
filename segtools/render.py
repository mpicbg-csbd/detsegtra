import numpy as np
from numba import jit


## the basic building block

@jit
def imgidx(img, idx):
  s = list(img.shape)[1:]
  res = np.zeros(s)
  for i in range(idx.shape[0]):
      for j in range(idx.shape[1]):
          res[i,j] = img[idx[i,j],i,j]
  return res

## Rendering 3D stacks

def coord_first_nonzero(arr, axis, invalid_val=-1):
  """
  https://stackoverflow.com/questions/47269390/numpy-how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array?rq=1
  """
  mask = arr!=0
  ## only if rgb array
  if arr.shape[-1]==3:
      mask = (arr!=0).all(-1)
  ids = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
  res = imgidx(arr, ids)
  return res

def decay(arr, rate=0.02):
  "as rate goes to 0 decay goes to max projection. only along z-dimension. higher z-index = deeper tissue."
  dec  = np.exp(np.arange(arr.shape[0])*rate)
  arr2 = arr*dec.reshape([-1,1,1])
  ids = arr2.argmax(0)
  res = imgidx(arr, ids)
  return res




# def max_proj_z_color(arr,plt):
#   ids = arr.argmax(0)
#   cmap0 = plt.get_cmap('viridis')
#   cmap1 = plt.get_cmap('Greys')
#   plt.imshow(arr.max(0), cmap=cmap1, alpha=1)
#   plt.imshow(ids, cmap=cmap0, alpha=1)

## designed for fly

# def fourpanel(img):
#   top = twoview(img, axis=2, stackaxis=0, alpha=0.7)
#   perm = np.arange(img.ndim)
#   perm[0] = 2; perm[2] = 0
#   bot = twoview(img.transpose(perm), axis=2, stackaxis=0, alpha=0.7)
#   res = np.concatenate([top, bot], axis=0)
#   return res

# def twoview(fly, axis=2, stackaxis=1, alpha=1):
#   w = int(alpha*fly.shape[axis]//2)
#   sli = [slice(None, None)]*3
#   sli[axis] = slice(None, w)
#   left = fly[sli].max(axis)
#   sli[axis] = slice(-w, None)
#   right = fly[sli].max(axis)
#   # if axis!=0:
#   #   left = zoom(left, (5,1)).T
#   #   right = zoom(right, (5,1)).T
#   res = np.concatenate((left, right), axis=stackaxis)
#   return res

## other projections

# def max_three_sides(stack,axis=None):
#   "stack has axes 'ZYX'"
#   assert stack.shape[0] == stack.shape[1] == stack.shape[2]
#   yx = stack.max(0)
#   zx = stack.max(1)
#   zy = stack.max(2)
#   if axis is None:
#     return np.array([yx,zx,zy])
#   else:
#     return np.concatenate([yx,zx,zy],axis)

## image-indexing

