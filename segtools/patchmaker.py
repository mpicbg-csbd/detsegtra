from math import ceil, floor
import numpy as np
import itertools

## utils: converts starting/ending index pairs into lists of slices

def se2slices(s,e):
    return [slice(s[j],e[j]) for j in range(len(s))]

def starts_ends_to_slices(starts,ends):
    s0 = np.array(starts.shape)
    if starts.ndim > 2:
        starts = starts.reshape([s0[0], s0[1:].prod()]).T
        ends = ends.reshape([s0[0], s0[1:].prod()]).T
        print(starts.shape, ends.shape)
    n = starts.shape[0]
    return [se2slices(starts[i], ends[i]) for i in range(n)]

## starts and ends api

def patchtool(stuff_we_know):
    """
    takes a dictionary of constraints on the patch structure.
    Returns as much as we can given those constraints.
    """
    result = dict()
    s = stuff_we_know
    keyset = set(s.keys())

    def linspace(w, n):
        l = np.linspace(0,w,n)
        l += 0.5
        l = np.floor(l).astype(np.int)
        return l

    def heterostride(domain, npts):
        starts = [linspace(domain[i], npts[i]) for i in range(len(domain))]
        starts = np.array(list(itertools.product(*starts)))
        return starts

    n=len(s.get('img',s.get('grid')))
    for k,v in s.items():
        if type(v) in [tuple,list,np.array]:
            s[k] = np.array(v)
        else:
            s[k] = np.array((v,)*n)

    # locals().update(stuff_we_know)

    if   keyset == {'grid', 'stride'}:
        # n = len(s['grid'])
        inds = np.indices(s['grid']).T.reshape([-1,n])
        starts = inds * s['stride']
        result['starts'] = starts
        result['inds'] = inds
    elif keyset == {'img', 'patch', 'overlap_factor'}:
        s['grid'] = np.ceil(s['img']/s['patch']*s['overlap_factor']).astype(np.int)
        inds = np.indices(s['grid']).T.reshape([-1,n])
        starts = heterostride(s['img'] - s['patch'], s['grid'])
        ends = starts + s['patch']
        result['starts'] = starts
        result['ends'] = ends
        result['slices'] = starts_ends_to_slices(starts, ends)
        result['inds'] = inds
    elif keyset == {'img', 'patch', 'stride'}:
        s['grid'] = np.ceil(s['img']/s['patch']).astype(np.int)
        inds = np.indices(s['grid']).T.reshape([-1,n])
        starts = inds * s['stride']
        # starts = heterostride(s['img'] - s['patch'], s['grid'])
        ends = starts + s['patch']
        result['starts'] = starts
        result['ends'] = ends
        result['slices'] = starts_ends_to_slices(starts, ends)
        result['inds'] = inds
    elif keyset == {'img', 'patch', 'grid'}:
        starts = heterostride(s['img'] - s['patch'], s['grid'])
        inds = np.indices(s['grid']).T.reshape([-1,n])
        result['inds'] = inds
        ends = starts + s['patch']
        result['starts'] = starts
        result['ends'] = ends
        result['slices'] = starts_ends_to_slices(starts, ends)
    elif keyset == {'img', 'patch', 'borders'}:
        patch_valid = s['patch'] - 2*s['borders']
        grid = np.ceil(s['img']/patch_valid).astype(np.int)
        starts_valid = heterostride(s['img'] - patch_valid, grid)
        ends_valid = starts_valid + patch_valid
        starts_padded = starts_valid
        ends_padded = starts_padded + s['patch']
        inds = np.indices(grid).T.reshape([-1,n])
        result['inds'] = inds
        result['starts_padded'] = starts_padded
        result['ends_padded'] = ends_padded
        result['slices_valid']  = starts_ends_to_slices(starts_valid, ends_valid)
        result['slices_padded'] = starts_ends_to_slices(starts_padded, ends_padded)
        result['slice_patch']   = se2slices(starts_valid[0]+s['borders'], ends_valid[0]+s['borders'])
    elif keyset == {'img', 'patch', 'borders','stride_factor'}:
        patch_valid = s['patch'] - 2*s['borders']
        sf = s['stride_factor']
        assert (patch_valid/sf == patch_valid//sf).all()
        grid = np.ceil(s['img']/patch_valid).astype(np.int)
        border2 = grid*patch_valid - s['img']
        inds = np.indices(grid).T.reshape([-1,n])
        starts_valid = inds * patch_valid
        # starts_valid = starts + s['borders']
        # starts_valid = heterostride((s['img'] - patch_valid)/sf, grid)*sf
        ends_valid = starts_valid + patch_valid
        starts_padded = starts_valid
        ends_padded = starts_padded + s['patch']
        # inds = np.indices(grid).T.reshape([-1,n])
        result['inds'] = inds
        result['border2'] = border2
        result['starts_padded'] = starts_padded
        result['ends_padded'] = ends_padded
        result['slices_valid']  = starts_ends_to_slices(starts_valid, ends_valid)
        result['slices_padded'] = starts_ends_to_slices(starts_padded, ends_padded)
        result['slice_patch']   = se2slices(starts_valid[0]+s['borders'], ends_valid[0]+s['borders'])
        padding = [(s['borders'][i],border2[i]+s['borders'][i]) for i in range(s['img'].ndim-1)]
        slice_orig   = [slice(s['borders'][i],-border2[i]-s['borders'][i]) for i in range(s['img'].ndim-1)]
        slice_orig   = tuple(slice_orig)
        result['padding'] = padding
        result['slice_orig'] = slice_orig
        slice_borders2   = [slice(0,-border2[i]) for i in range(s['img'].ndim-1)]
        slice_borders2   = tuple(slice_borders2)
        result['slice_borders2'] = slice_borders2
    elif keyset == {'img', 'patch', 'borders', 'overlap_factor'}:
        patch_valid = s['patch'] - 2*s['borders']
        grid = np.ceil(s['img']/patch_valid*s['overlap_factor']).astype(np.int)
        starts_valid = heterostride(s['img'] - patch_valid, grid)
        ends_valid = starts_valid + patch_valid
        starts_padded = starts_valid
        ends_padded = starts_padded + s['patch']
        inds = np.indices(grid).T.reshape([-1,n])
        result['inds'] = inds
        result['starts_padded'] = starts_padded
        result['ends_padded'] = ends_padded
        result['slices_valid']  = starts_ends_to_slices(starts_valid, ends_valid)
        result['slices_padded'] = starts_ends_to_slices(starts_padded, ends_padded)
        result['slice_patch']   = se2slices(starts_valid[0]+s['borders'], ends_valid[0]+s['borders'])
    else:
        print("ERROR: Your keys are not a valid patch request.")

    return result

## testing

def test_patchtool():
  test = np.zeros((100,100))
  container = np.zeros(test.shape)
  borders = (4,5)
  res = patchtool({'sh_img':test.shape, 'sh_patch':(32,32), 'sh_borders':borders})

  padding = np.array([borders, borders]).T
  test = np.pad(test, padding, mode='constant')
  print(test.shape)
  s2 = res['slice_patch']

  for i in range(len(res['slices_valid'])):
    s1 = res['slices_padded'][i]
    s3 = res['slices_valid'][i]
    x  = test[s1]
    # x = x / x.mean((1,2,3))
    print(x.shape)
    print(s1, s2, s3)
    container[s3] += 1

  return container

def coverage(imgshape, slices):
    coverage = np.zeros(imgshape)
    for ss in slices:
        coverage[ss] += 1
    return coverage, {'mean':coverage.mean(), 'uniq':np.unique(coverage)}

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

## other utils

def perfect_padding(imgsize, patchsize, minpad=None):
    n = len(patchsize)
    mods = [imgsize[i] % patchsize[i] for i in range(n)]
    p = patchsize
    m = mods

    def fixmin(x,i):
        while x < minpad[i]:
            x += patchsize[i]
        return x

    def f(i):
        if m[i]==0:
            l,r = 0,0
        else:
            l,r = (p[i] - m[i]//2, p[i] - (m[i]-m[i]//2))
        l = fixmin(l,i)
        r = fixmin(r,i)
        return l,r

    pads = [f(i) for i in range(n)]
    dif = len(imgsize) - len(pads)
    if dif > 0: pads += [(0,0)]*dif
    return pads

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

## Thu Jul  5 15:41:34 2018

combine slices_perfect_covering and slices_grid into one function.
    works for any dimension.
    works as always perfect covering OR homogeneous patch size mode.

How should we approach tiled triplets?
Should it also have the option for enforcing homogeneous patch size vs perfect covering?
Should this call slices_grid under the hood? should we let the user call slices grid and we just turn
slices grid into a list of appropriate slice triplets? Yes, this gives us direct control over slices
without having to pass params deep down into functions.

Sat Jul  7 13:34:57 2018

Here's how I think about slices and padding currently.

1. Don't worry about border effects of applied functions when making slices.
Slices should be allowed to overlap. We should be able to control slice shape and stride
independently. Slices with fixed stride must also return a list of ints for the extra pixels covered
by the patches in each dimension. Can be positive if stride extends beyond image bounds or negative
if stride doesn't reach image edge. With perfect grid / const striding a single int is sufficient to 
describe and entire dimension. This list of ints, if positive, could be used to add padding to the
end of each image dimension in order to allow patches of constant size.
Another way of getting patches of constant shape while still covering every pixel at least once
is to allow the stride to vary across the image. We might want that the stride is constant until the
very last slice, which then overlaps heavily with the previous slice, or we might want that the stride
varies a small amount between each consecutive pair of slices.

desireable properties:
- no need to pad twice
- slice properties hold immediately, no need to pad *afterwards*.
- this means we *have* to pad correctly and properly *before* we compute slices, or join the
two operations.

- note: memory could increase dramatically if we have a large sliceshape, small stride and a large boundary.
- to avoid memory increase we could only feed in slices via a generator.

- the ability to avoid applying padding twice.
Ideally we would wouldn't have to first pad the image, then compute slices, then add extra padding so
that the slices actually fit perfectly into the image.
- separate functions which return slices of constant shape from those that have varied shape.
    - If we want the total coverage of the image to be exactly 1 everywhere and we're not willing
    to pad the image, then we must allow for heterogeneous slice shapes.
    - If we want coverage==1 everywhere and we want homogeneous slice shapes, then we must pad.
    - We want these properties to hold *immediately*, not just *after we pad the image*.
    - This means the function should also do the padding for us.
    - If we're concerned that padding requires too much memory then we can build a generator that
    returns only ndarrays of a fixed shape, and that maps a slice of the array to a slice of the full image.

use case:
- I want to apply a convnet to an image without seeing border effects in the middle of the image.
- I want to build training data xs,ys and ws.
    - 2D case where I want stride < patchshape and i want to collapse z dim into channels.
        - Here i also want a small amount of overlap between patches.
    - 3D case where I want some overlap between neighboring patches dues to boundary effects.

Mon Jul 16 13:08:02 2018

Under what conditions is the set of starting / ending indices well defined?
- stride and gridshape defines set of starts
- imgshape and stride 
- imgshape and gridshape... gridshape * stride = imgshape

"""



