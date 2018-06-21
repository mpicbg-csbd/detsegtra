from math import ceil, floor
import numpy as np
import itertools


## high level operations. use slices lists under the hood.

def apply_tiled_kernel(func, arr, border):
    border = np.array(border)
    assert len(border)==arr.ndim
    arrshape = np.array(arr.shape)
    out = np.zeros(arrshape - border, dtype=arr.dtype)
    outshape = np.array(out.shape)
    slices = slices_perfect_covering(outshape, 2*border)
    for ss in slices:
      ss2 = translate(ss, border)
      ss2 = grow(ss2, border)
      res = func(arr[ss2])
      ss3 = grow(slice_from_shape(res.shape), -border)
      out[ss] = res[ss3]
    return out

## return lists of slices or slice tuples

def slices_perfect_covering(imgshape, sliceshape):
    "slices at end of each dimension may have smaller size."
    if not hasattr(sliceshape,'__len__'):
        sliceshape = [sliceshape] * len(imgshape)

    def f(i): 
        l = list(range(0,imgshape[i],sliceshape[i])) + [imgshape[i]]
        l2 = [slice(l[j], l[j+1]) for j in range(len(l)-1)]
        return l2

    slices = list(itertools.product(*[f(i) for i in range(len(imgshape))]))
    return slices

def tiled_triplets(shape_unpadded, sliceshape, border):
    "return list of slice triplets: input, output and container. for tiling operations."
    border = np.array(border)
    shape_unpadded = np.array(shape_unpadded)
    sliceshape = np.array(sliceshape)

    assert len(border)==len(shape_unpadded)
    slices = slices_perfect_covering(shape_unpadded, sliceshape)
    def f(ss_container):
      ss_input = translate(ss_container, border)
      ss_input = grow(ss_input, border)
      sh = np.array(shape_from_slice(ss_container))
      ss_output = translate(slice_from_shape(sh), border)
      return (ss_input, ss_output, ss_container)
    triplets = [f(ss) for ss in slices]
    return triplets

def slices_grid(imgshape, sliceshape, overlap=(0,0,0), offset=(0,0,0)):
    "slices do no not go beyond boundaries. boundary conditions must be handled separately."

    if not hasattr(sliceshape,'__len__'):
        sliceshape = [sliceshape] * len(imgshape)
    if not hasattr(overlap,'__len__'):
        overlap = [overlap] * len(imgshape)
    if not hasattr(offset,'__len__'):
        offset = [offset] * len(imgshape)

    def f(i,n):
        return slice(i,i+sliceshape[n])

    def g(i):
        return (offset[i], imgshape[i]-sliceshape[i]+1, sliceshape[i]-overlap[i])

    alist = np.arange(*g(0))
    blist = np.arange(*g(1))
    if len(imgshape)==2:
        it = itertools.product(alist, blist)
        slices = [[f(i,0), f(j,1)] for i,j in it]
    elif len(imgshape)==3:
        clist = np.arange(*g(2))
        it = itertools.product(alist, blist, clist)
        slices = [[f(i,0), f(j,1), f(k,2)] for i,j,k in it]
    
    return np.array(slices)

## operations on slices

def grow(ss, dx=1):
    if not hasattr(dx,'__len__'):
        dx = [dx]*len(ss)
    ss = list(ss)
    for i,sl in enumerate(ss):
        a = sl.start - dx[i]
        b = sl.stop + dx[i]
        assert 0 <= a < b
        ss[i] = slice(a,b,sl.step)
    return ss

def translate(ss, dx=0):
    if not hasattr(dx,'__len__'):
        dx = [dx]*len(ss)
    ss = list(ss)
    for i,sl in enumerate(ss):
        a = sl.start + dx[i]
        b = sl.stop + dx[i]
        assert 0 <= a < b
        ss[i] = slice(a,b,sl.step)
    return ss

## build single slices

def centered_slice(centerpoint, w=30):
  ndim = len(centerpoint)
  if not hasattr(w,'__len__'):
    w = [w,]*ndim
  def sd(i):
    return slice(floor(centerpoint[i]-w[i]), floor(centerpoint[i]+w[i]))
  ss = [sd(i) for i in range(ndim)]
  return ss

## convert to/from shape

def shape_from_slice(ss):
    "note this is not equivalent to img[ss].shape if ss has negative values or step!=1."
    return tuple([sl.stop-sl.start for sl in ss])

def slice_from_shape(shape):
  return [slice(0, s) for s in shape]

## tiling utils, don't actually build slices lists.

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

def compute_subblocks(shape, blocksize=100):
    "for use with the sub_blocks parameter in gputools"
    if not hasattr(blocksize,'__len__'):
        blocksize = [blocksize] * len(shape)
    def f(i): return ceil(shape[i] / blocksize[i])
    return [f(i) for i in range(len(shape))]

## deprecated. use slices instead of patches.

@DeprecationWarning
def sample_patches(data, patch_size, n_samples=100, verbose=False):
    """
    sample 2d patches of size patch_size from data
    """
    assert np.all([s <= d for d, s in zip(data.shape, patch_size)])
    # change filter_mask to something different if needed
    filter_mask = np.ones_like(data)
    # get the valid indices
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, data.shape)])
    valid_inds = np.where(filter_mask[border_slices])
    if len(valid_inds[0]) == 0:
        raise Exception("could not find anything to sample from...")
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    # sample
    sample_inds = np.random.randint(0, len(valid_inds[0]), n_samples)
    rand_inds = [v[sample_inds] for v in valid_inds]
    res = np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2, r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2] for r in zip(*rand_inds)])
    return res

@DeprecationWarning
def sample_patches_from_img(coords, img, shape, boundary_cond='mirror'):
    """
    get patches from an image given coordinates and patch shapes.
    TODO: enable boundary conditions on all sides of the img, not just bottom and right.
    """
    y_width, x_width = shape
    # assert coords[:,0].max() <= img.shape[0]-x_width
    # assert coords[:,1].max() <= img.shape[1]-y_width
    if boundary_cond=='mirror':
        a,b = img.shape
        img2 = np.zeros((2*a, 2*b), dtype=img.dtype)
        img2[:a, :b] = img.copy()
        img2[a:2*a, :b] = img[::-1,:].copy()
        img2[:a,b:2*b] = img[:,::-1].copy()
        img2[a:2*a, b:2*b] = img[::-1, ::-1].copy()
        img = img2
    patches = np.zeros(shape=(coords.shape[0], x_width, y_width), dtype=img.dtype)
    for m,ind in enumerate(coords):
        patches[m] = img[ind[0]:ind[0]+x_width, ind[1]:ind[1]+y_width]
    return patches

@DeprecationWarning
def random_patch_coords(img, n, shape):
    "Different ways of sampling pixel coordinates from an image"
    y_width, x_width = shape
    xc = np.random.randint(img.shape[0]-x_width, size=n)
    yc = np.random.randint(img.shape[1]-y_width, size=n)
    return np.stack((xc, yc), axis=1)

@DeprecationWarning
def regular_patch_coords(img, patchshape, step):
    "deprecated because we don't want coordinates to depend on patchshape"
    coords = []
    dy, dx = img.shape[0]-patchshape[0], img.shape[1]-patchshape[1]
    for y in range(0,dy,step):
        for x in range(0,dx,step):
            coords.append((y,x))
    return np.array(coords)

@DeprecationWarning
def square_grid_coords(img, step):
    a,b = img.shape
    a2,ar = divmod(a, step)
    b2,br = divmod(b, step)
    a2 += 1
    b2 += 1
    ind = np.indices((a2, b2))
    ind *= step
    ind = np.reshape(ind, (2, a2*b2))
    ind = np.transpose(ind)
    return ind

@DeprecationWarning
def piece_together(patches, coords, imgshape=None, border=0):
    """
    piece together a single image from a list of coordinates and patches
    patches must all be same shape!
    patches.shape = (sample, x, y, channel) or (sample, x, y)
    coords.shape  = (sample, 2)
    TODO: potentially add more ways of recombining than a simple average, i.e. maximum, etc
    """

    if patches.ndim == 3:
        patches = patches[:,:,:,np.newaxis]
    n_samp, dx, dy, channels = patches.shape
    
    x_size = coords[:,0].max() + dx
    y_size = coords[:,1].max() + dy
    if imgshape:
        x_host, y_host = imgshape
        x_size, y_size = max(x_size, x_host), max(y_size, y_host)
    patch_img = np.zeros(shape=(x_size, y_size,channels))
    count_img = np.zeros(shape=(x_size, y_size,channels))

    # ignore parts of the image with boundary effects
    mask = np.ones((dx, dy, channels))
    if border>0:
        mask[:,0:border] = 0
        mask[:,-border:] = 0
        mask[0:border,:] = 0
        mask[-border:,:] = 0

    for cord, patch in zip(coords, patches):
        x,y = cord
        patch_img[x:x+dx, y:y+dy] += patch*mask
        count_img[x:x+dx, y:y+dy] += np.ones_like(patch)*mask

    # if imgshape:
    #     a,b = imgshape
    #     patch_img = patch_img[:a,:b]
    #     count_img = count_img[:a,:b]
    
    res = patch_img/count_img
    if imgshape:
        a,b = imgshape
        res = res[:a, :b]
    return res

@DeprecationWarning
def piece_together_ragged_2d(patches, coords):
    """
    patches is a list of ndarrays of potentially varying size.
    coords is an ndarray of shape Nx2
    """

    def extent(i):
        dx,dy = patches[i].shape
        x,y = coords[i]
        return x+dx, y+dy

    xs, ys = zip(*[extent(i) for i in range(coords.shape[0])])
    x_size, y_size = max(xs), max(ys)
    patch_img = np.zeros(shape=(x_size, y_size))
    for i in range(coords.shape[0]):
        img = patches[i]
        dx,dy = img.shape
        x,y = coords[i]
        patch_img[x:x+dx, y:y+dy] = img
    return patch_img

@DeprecationWarning
def sub_block_apply(func, img, sub_blocks=(1,1,1)):
    """
    apply a function to subblocks of a 3D stack
    """
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


history = """

## Wed Jun 20 12:54:35 2018

There are a variety of things we want to do with shape and slices.
Open problems include how to do simple tiling of operations over large images in a way that respects boundary conditions.
For any operation we should be able to specify a boundary width beyond which the boundary cannot be felt.
For purely convolutional operations it should be possible to apply them to large images without creating visible seams / tiling artifacts.
Padding should be done by np.pad, outside of our function. But our function must know the boundary width of the operation to be applied!
we should have a list of slices triplets:
    1. enlarged slice for the padded input
    2. slice into valid region of output
    3. slice with same shape as 2 mapping into result container.
        is container padded? we must decide. let's say no.
    slices do not all need to be same size!
`tiled_triplets` does this!

## Thu Jun 21 18:51:05 2018

rearrange function order and group into sections.



"""