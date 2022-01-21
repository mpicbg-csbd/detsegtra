import numpy as np
from numba import jit
from scipy import linalg as LA


# taken from stackoverflow
def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # import numpy as np
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

def moments_simple_2nd(img):
    mu = np.zeros((3,)*3)
    mu[0, 0, 0] = np.sum(img)
    ind = np.indices(img.shape)
    mu[1, 0, 0] = np.sum(ind[0]*img)
    mu[0, 1, 0] = np.sum(ind[1]*img)
    mu[0, 0, 1] = np.sum(ind[2]*img)
    mu[1, 1, 0] = np.sum(ind[0]*ind[1]*img)
    mu[0, 1, 1] = np.sum(ind[1]*ind[2]*img)
    mu[1, 0, 1] = np.sum(ind[0]*ind[2]*img)
    mu[1, 1, 1] = np.sum(ind[0]*ind[1]*ind[2]*img)
    return mu

# @jit('f4[:,:,:](f4[:,:,:],f4[:],u1)')
@jit
def moments_central(image, cen, max_order):
    # cdef Py_ssize_t p, q, r, c
    mu = np.zeros((max_order + 1, max_order + 1, max_order + 1), dtype=np.double)
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
                for p1 in range(max_order + 1):
                    dcx2 = 1
                    for p2 in range(max_order + 1):
                        dcx3 = 1
                        for p3 in range(max_order + 1):
                            mu[p1, p2, p3] += val * dcx1 * dcx2 * dcx3
                            dcx3 *= x3
                        dcx2 *= x2
                    dcx1 *= x1
    return np.asarray(mu)

def inertia_tensor(mu):
    """
    mu = moments_central
    """
    inertia_tensor = [[mu[2, 0, 0], mu[1, 1, 0], mu[1, 0, 1]],
                      [mu[1, 1, 0], mu[0, 2, 0], mu[0, 1, 1]],
                      [mu[1, 0, 1], mu[0, 1, 1], mu[0, 0, 2]]]
    inertia_tensor = np.array(inertia_tensor)
    inertia_tensor /= mu[0, 0, 0]
    return inertia_tensor

def ellipse(n=100, z=[1, 1, 1]):
    cutoff = 3*n
    img = np.zeros((n, n, n), np.float)
    elli = np.indices(img.shape, np.float)
    elli -= np.array([(n-1)/2, (n-1)/2, (n-1)/2]).reshape((3, 1, 1, 1))
    elli *= np.array(z).reshape((3, 1, 1, 1))
    elli = np.sum(elli**2, axis=0)
    mask = (elli < cutoff).astype(np.float)
    return mask, elli

def lap_of_gaus_nd(x, sig=6):
    "x is an nd-vector in Real Euclidean space."
    e = np.e
    π = np.pi
    σ = sig

    r2 = (x*x).sum()/(2*σ**2)
    m1 = 1 / (π * σ**4)
    m2 = 1 - r2
    m3 = e**(-r2)
    m4 = m1*m2*m3
    return m4

def kernel_log_3d(sig=2, w=10):
    "normed s.t. res.sum()==1"
    dom = np.indices((w,)*3)
    dom = dom - (w-1)/2
    res = [lap_of_gaus_nd(x, sig) for x in dom.reshape(3, -1).T]
    res = np.array(res).reshape(dom.shape[1:])
    res = res/res.sum()
    return res

def kernel_log_3d_2(sig=2,w=10):
    "an example of building a kernel using `build_kernel_nd`."
    func = lambda x: lap_of_gaus_nd(x, sig)
    kern = build_kernel_nd(w, 3, func)
    kern = kern/kern.sum()
    return kern

def autocorrelation(x):
  """
  nD autocorrelation
  remove mean per-patch (not global GT)
  normalize stddev to 1
  value at zero shift normalized to 1...
  """
  x = (x - np.mean(x))/np.std(x)
  x  = np.fft.fftn(x)
  x  = np.abs(x)**2
  x = np.fft.ifftn(x).real
  x = x / x.flat[0]
  x = np.fft.fftshift(x)
  return x


## deprecated

@DeprecationWarning
def centered_kernel_nd(func, w=[10,10,10]):
  """
  Deprecated in favor of technique like the following:

  sh = (20,20,20)
  def f(x):
    x = x - np.array(sh)/2
    x = x/4
    r = np.sqrt(x[1]**2 + x[2]**2)
    z = np.abs(x[0])
    k = 1.0 + 1.0J
    return airy(r)[0] * np.exp(-k*z).real
  kern = np.array([f(x) for x in np.indices(sh).reshape((len(sh),-1)).T]).reshape(sh)

  centered, equal-sided kernel with side-length w
  uses any func : R^n -> R
  does not normalize
  """
  w = np.array(w)
  dom = np.indices(w) # domain
  dom = dom - (w[:,None,None,None]-1)/2
  res = [func(x) for x in dom.reshape(len(w),-1).T]
  res = np.array(res).reshape(dom.shape[1:])
  return res

@DeprecationWarning
def place_gauss_at_pts(pts, w=[6,6,6]):
  # assert w[0]%1==0
  w  = np.array(w)
  w6 = np.ceil(w).astype(np.int)*6 + 1 ## gaurantees odd size and thus unique, brightest center pixel
  def f(x):
    return np.exp(-(x*x/w/w).sum()/2)
  kern = centered_kernel_nd(f,w6)
  kern = kern / kern.max()
  res = conv_at_pts(pts, kern)
  return res,kern

def se2slice(s,e):
  # def f(x): return x if x is not in [0,-0] else None
  return tuple(slice(a,b) for a,b in zip(s,e))



## place gaussians within 3D volume

@DeprecationWarning
def place_gauss_at_pts2(pts, sigma=[6,6,6], kern=[37,37,37]):
  "place gaussians at specified points. with center of Gaussian located at the speicified point. WARNING. now we use conv_at_pts directly."
  # assert w[0]%1==0
  s  = np.array(sigma)
  kern = np.array(kern)
  sh = np.where(kern%2==1,kern,kern+1)  ## gaurantees odd size and thus unique, brightest center pixel
  def f(x):
    x = x - sh/2
    return np.exp(-(x*x/s/s).sum()/2)
  kern = np.array([f(x) for x in np.indices(sh).reshape((len(sh),-1)).T]).reshape(sh)
  kern = kern / kern.max()
  res  = conv_at_pts(pts, kern)
  return res,kern

### deps

@DeprecationWarning
def conv_at_pts3(pts,kern,sh):
  """
  Fast convolution for sparse image (described with 1's at pts) and _sparse_ kern.
  """
  kern = np.zeros((19,3,3)) ## must be odd
  kern[:,1,1] = 1
  kern[9] = 1
  kern[9,1,1] = 2

  ma = kern!=0
  ks = np.array(kern.shape)
  pts_kern = np.indices(ks)
  target = np.zeros(ks + sh - (1,1,1))
  pts = (np.random.rand(int(np.prod(sh)*0.01),3)*sh).astype(int)
  for p,v in zip(pts_kern[:,ma].T,kern[ma]):
    target[tuple((pts + p).T)] = v # np.maximum(v,target[tuple((pts + p).T)])
  a,b,c = ks // 2
  target = target[a:-a,b:-b,c:-c]
  target = (target>=1).astype(np.uint8)


@DeprecationWarning
def conv_at_pts2(pts,kern,sh,func=lambda a,b:a+b):
  "kernel is centered on pts. kern must have odd shape. sh is shape of output array."
  assert pts.ndim == 2;
  assert kern.ndim == pts.shape[1] == len(sh) == 3

  ks = np.array(kern.shape)
  assert np.all(ks%2==1)
  a,b,c = ks//2

  output = np.zeros(ks + sh - (1,1,1))
  for p in pts:
    # z,y,x = p
    ss = se2slice(p,p+ks)
    output[ss] = func(output[ss],kern)
  output = output[a:-a,b:-b,c:-c]
  return output

# TODO: implement this
# def place_gaussian_at_pts_subpixel(pts,s=[3,3],ks=[63,63],sh=[64,64]):
#   """
#   s  = sigma for gaussian
#   ks = kernel size
#   sh = target/container shape
#   """
#   s  = np.array(s)
#   ks = np.array(ks)
# 
#   def f(x):
#     x = x - (ks-1)/2
#     return np.exp(-(x*x/s/s).sum()/2)
#   kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
#   kern = kern / kern.max()
#   target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
#   return target

@DeprecationWarning
def place_gaussian_at_pts(pts,sigmas=[3,3],shape=[64,64]):
  """
  sigmas  = sigma for gaussian
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

# def place_gaussian_at_pts(pts,sh=(,sigmas):
#   s  = np.array(sigmas)
#   ks = (s*7).astype(np.int) ## must be ODD
#   def f(x):
#     x = x - (ks-1)/2
#     return np.exp(-(x*x/s/s).sum()/2)
#   kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
#   kern = kern / kern.max()
#   target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
#   return target


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

def conv_at_pts_multikern(pts,kerns,sh,func=lambda a,b:np.maximum(a,b),beyond_borders=False):
  
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

  # print("min extent: ", min_extent)
  # print("max extent: ", max_extent)

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


## coordinate transforms

def cart2pol(x, y):
  rho = np.sqrt(x**2 + y**2)
  phi = np.arctan2(y, x)
  return(rho, phi)

def pol2cart(rho, phi):
  x = rho * np.cos(phi)
  y = rho * np.sin(phi)
  return(x, y)

def xyz2rthetaphi(v):
  x,y,z = v
  r = np.sqrt(x**2 + y**2 + z**2)
  th = np.arctan(y/x)
  ph = np.arccos(z/r)
  return r,th,ph
