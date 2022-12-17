import numpy as np
from .math_utils import se2slice
from itertools import product
from types import SimpleNamespace

import ipdb

def place_gaussian_at_pts(pts,sigmas=[3,3],shape=[64,64]):
  """
  sigmas = sigma for gaussian
  shape = target/container shape
  """
  s  = np.array(sigmas)
  ks = (7*s).astype(int)
  ks = ks - ks%2 + 1## enfore ODD shape so kernel is centered! (grow even dims by 1 pix)
  sh = shape

  def f(x):
    x = x - (ks-1)/2
    return np.exp(-(x*x/s/s).sum()/2)
  kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  kern = kern / kern.max()
  target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
  return target

def conv_at_pts4(pts,kern,sh,func=lambda a,b:a+b):
  "kernel is centered on pts. kern must have odd shape. sh is shape of output array."
  assert pts.ndim == 2;
  assert kern.ndim == pts.shape[1] == len(sh)
  assert 'int' in str(pts.dtype)

  kerns = [kern for _ in pts]
  return conv_at_pts_multikern(pts,kerns,sh,func)

def test_conv_at_pts_multikern_3d():
  pts = np.random.rand(100,3)*(30,100,500)
  pts = pts.astype(np.int)
  kern_shapes = np.random.rand(100,3)*(3,3,3) + (11,12,13)
  kern_shapes = kern_shapes.astype(np.int)
  kerns = [np.indices(k).sum(0) for i,k in enumerate(kern_shapes)]
  res = conv_at_pts_multikern(pts,kerns,(30,100,500),lambda a,b:a+b)
  return res

def test_conv_at_pts_multikern_2d():
  pts = np.random.rand(100,2)*(100,500)
  pts = pts.astype(np.int)
  kern_shapes = np.random.rand(100,2)*(3,3) + (11,12)
  kern_shapes = kern_shapes.astype(np.int)
  kerns = [np.indices(k).sum(0) for i,k in enumerate(kern_shapes)]
  res = conv_at_pts_multikern(pts,kerns,(100,500),lambda a,b:a-b)
  return res

def test_conv_at_pts_multikern_1d():
  pts = np.random.rand(100,1)*(500,)
  pts = pts.astype(np.int)
  kern_shapes = np.random.rand(100,1)*(3,) + (31,)
  kern_shapes = kern_shapes.astype(np.int)
  kerns = [10*np.exp(-((np.indices(k).T+i-4)**2).sum(-1)) for i,k in enumerate(kern_shapes)]
  print(kerns[0], kern_shapes[0], kern_shapes.shape)
  print(pts.shape,len(kerns),kerns[0].shape)
  res = conv_at_pts_multikern(pts,kerns,(500,),lambda a,b:a+b)
  return res

def createTarget2D2(gt_pts, target_shape, sigmas):

  s  = np.array(sigmas)  # 3,3
  ks = (7*s).astype(int) # 21,21
  ks = ks - ks%2 + 1     ## enfore ODD shape so kernel is centered! (grow even dims by 1 pix) # 

  def f(x):
    x = x - (ks-1)/2
    return np.exp(-(x*x/s/s).sum()/2)
  kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  
  target = np.zeros(target_shape)
  w = ks//2 ## center coordinate

  ## max inbounds indices
  tx = np.array(target_shape)-1 # target
  kx = ks-1 # kernel

  for p in gt_pts:
    target_p0 = (p-w).clip(min=0)
    target_p1 = (p+w).clip(max=tx)
    kernel_p0 = (w-p).clip(min=0)
    kernel_p1 = (kx - (p+w - tx)).clip(max=kx)
    target_slice = tuple(slice(a,b+1) for a,b in zip(target_p0,target_p1))
    kernel_slice = tuple(slice(a,b+1) for a,b in zip(kernel_p0,kernel_p1))
  target[target_slice] = np.maximum(target[target_slice] , kern[kernel_slice])

  return target

def conv_at_pts_multikern(pts,kerns,sh,func=lambda a,b:np.maximum(a,b),beyond_borders=False):
  
  if len(kerns)==0: return np.zeros(sh)
  kern_shapes = np.array([k.shape for k in kerns])
  local_coord_center = kern_shapes//2
  min_extent  = (pts - local_coord_center).min(0).clip(max=[0]*len(sh))
  max_extent  = (pts - local_coord_center + kern_shapes).max(0).clip(min=sh)
  full_extent = max_extent - min_extent
  pts2 = pts - min_extent
  
  target = np.zeros(full_extent)

  for i,p in enumerate(pts2):
    ss = se2slice(p - local_coord_center[i], p - local_coord_center[i] + kern_shapes[i])
    target[ss] = func(target[ss], kerns[i])

  A = np.abs(min_extent)
  _tmp = sh-max_extent
  B = np.where(_tmp==0,None,_tmp)
  ss = se2slice(A,B)

  if beyond_borders is True:
    target2 = target.copy()
    target2[...] = -5
    target2[ss] = target[ss]
  else:
    target2 = target[ss]

  return target2


# from numba import jit
# @jit(nopython=True)
def createTarget2D(gt_pts, target_shape, sigmas):
  """ 
  gt_pts : Uint[n_cells,2] are discretized coordinates in `[0..target_shape)`
  target_shape : Uint[2]
  sigmas : Float[2] in downscaled space

  Using Numba for Just-In-Time (JIT) compilation gives 20x speedup.
  """ 

  target = zeroArray(target_shape)

  gs = sigmas # "gaussian sigmas"
  ks = floor(gs*7).astype(u32) # "kernel shape"
  ks = ks + 1 - ks%2 # ensure `ks%2==1` => unique central pixel

  # `p` iterates over centerpoint annotations and has pixel coordinates
  # Uint[2] within `target_shape` bounds.
  for p in gt_pts:

    # `k0` and `k1` iterate over single gaussian kernel with support `ks`
    for k0 in range(ks[0]): 
      for k1 in range(ks[1]):
        for k2 in range(ks[2]):

          # k in [-a..a]^2 inclusive. centered at 0 in each dim.
          k = array([k0,k1,k2]) - floor(ks / 2).astype(u32)

          # Sample Gaussian kernel at pixel `k`
          val = exp(-sum(k*k/gs/gs)/2)

          # `x` is pixel location in `target`
          x = p + k
          
          # bounds check
          if any(x<0) or any(x>=array(target_shape)): continue

          # use `max` not `sum` so kernels don't blur together on overlap
          target[x[0],x[1],x[2]] = max(val, target[x[0],x[1],x[2]])

  return target

array = np.array
indices = np.indices
zeros = np.zeros
maximum = np.maximum
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)


def createTarget(pts, target_shape, sigmas):
  s  = array(sigmas)
  ks = floor(7*s).astype(int)   ## extend support to 7/2 sigma in every direc
  ks = ks - ks%2 + 1            ## enfore ODD shape so kernel is centered! 

  ## create a single Gaussian kernel array
  def f(x):
    x = x - (ks-1)/2
    return exp(-(x*x/s/s).sum()/2)
  kern = array([f(x) for x in indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  
  target = zeros(ks + target_shape) ## include border padding
  w = ks//2                         ## center coordinate of kernel
  pts_offset = pts + w              ## offset by padding

  for p in pts_offset:
    target_slice = tuple(slice(a,b+1) for a,b in zip(p-w,p+w))
    target[target_slice] = maximum(target[target_slice], kern)

  remove_pad = tuple(slice(a,a+b) for a,b in zip(w,target_shape))
  target = target[remove_pad]

  return target

def testSplitIntoPatches(viewer):
  for w in [200,150,180,299,300,301]:
    for w0 in [22,24,32]:
      sw = (w,301)
      pw = (24,24)
      bw = (4,4)

      patches = splitIntoPatches(sw, pw, bw)

      inner = np.zeros(sw)
      outer = np.zeros(sw)
      for p in patches:
        inner[p.inner] += 1
        outer[p.outer] += 1
      ui = np.unique(inner) 
      uo = np.unique(outer)
      print(ui,uo)
      assert set(ui) == {1}
      assert set(uo) <= {1,2,4}

  # viewer.add_image(inner)
  # viewer.add_image(outer)

def splitIntoPatches(img_shape, outer_shape=(256,256), min_border_shape=(24,24)):
  """
  Split image into non-overlapping `inner` rectangles that exactly cover
  `img_shape`. Grow these rectangles by `border_shape` to produce overlapping
  `outer` rectangles to provide context to all `inner` pixels. These borders
  should be half the receptive field width of our CNN.

  CONSTRAINTS
  outer shape % 8 == 0 in XY (last two dims)
  img_shape   >= outer_shape 
  outer_shape >= 2*min_border_shape

  GUARANTEES
  outer shape == outer_shape forall patches
  inner shape <= outer_shape - min_border_shape
  acutal boder shape >= min_border_shape forall patches except on image bounds
  max(inner shape) - min(inner shape) <= 1 forall shapes
  
  """
  
  img_shape = array(img_shape)
  outer_shape = array(outer_shape)
  min_border_shape = array(min_border_shape)
  
  # make shapes divisible by 8
  assert all(outer_shape % 4 == 0), f"Error: `outer_shape` {outer_shape}%8 != 0."
  assert all(img_shape>=outer_shape), f"Error: `outer_shape` doesn't fit"
  assert all(outer_shape>=2*min_border_shape), f"Error: borders too wide"

  # our actual shape will be <= this desired shape. `outer_shape` is fixed,
  # but the border shape will grow.
  desired_inner_shape = outer_shape - min_border_shape

  ## round up, shrinking the actual inner_shape
  inner_counts = ceil(img_shape / desired_inner_shape)
  inner_shape_float = img_shape / inner_counts

  def f(i,n):
    a = inner_shape_float[n]
    b = floor(a*i)      # slice start
    c = floor(a*(i+1))  # slice stop
    inner = slice(b,c)
    
    w = c-b

    # shift `outer` patch inwards when we hit a border to maintain outer_shape.
    if b==0:                          # left border
      outer = slice(0,outer_shape[n])
      inner_rel = slice(0,w)
    elif c==img_shape[n]:             # right border 
      outer = slice(img_shape[n]-outer_shape[n],img_shape[n])
      inner_rel = slice(outer_shape[n]-w,outer_shape[n])
    else:
      r = b - min_border_shape[n]
      outer = slice(r,r+outer_shape[n])
      inner_rel = slice(min_border_shape[n],min_border_shape[n]+w)
    return SimpleNamespace(inner=inner, outer=outer, inner_rel=inner_rel)

  ndim = len(inner_counts)
  slices_lists = [[f(i,n) for i in range(inner_counts[n])] 
                          for n in range(ndim)]

  def g(s):
    inner = tuple(x.inner for x in s)
    outer = tuple(x.outer for x in s)
    inner_rel = tuple(x.inner_rel for x in s)
    return SimpleNamespace(inner=inner, outer=outer, inner_rel=inner_rel)

  # equivalent to itertools.product(*slices_lists)
  # prod = array(np.meshgrid(*slices_lists)).reshape((ndim,-1)).T

  res = [g(s) for s in product(*slices_lists)]
  return res























