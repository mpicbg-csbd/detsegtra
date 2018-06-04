import numpy as np
from numba import jit


## Rendering

def first_nonzero(arr, axis, invalid_val=-1):
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

def max_proj_z_color(arr,plt):
  ids = arr.argmax(0)
  cmap0 = plt.get_cmap('viridis')
  cmap1 = plt.get_cmap('Greys')
  plt.imshow(arr.max(0), cmap=cmap1, alpha=1)
  plt.imshow(ids, cmap=cmap0, alpha=1)



def decay(arr, rate=0.02):
  "as rate goes to 0 decay goes to max projection. only along z-dimension. higher z-index = deeper tissue."
  dec  = np.exp(np.arange(arr.shape[0])*rate)
  arr2 = arr*dec.reshape([-1,1,1])
  ids = arr2.argmax(0)
  res = imgidx(arr, ids)
  return res

## image-indexing

@jit
def imgidx(img, idx):
  res = np.zeros(idx.shape + (3,))
  for i in range(idx.shape[0]):
      for j in range(idx.shape[1]):
          res[i,j] = img[idx[i,j],i,j]
  return res