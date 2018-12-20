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

def build_kernel_nd(func, w=[10,10,10]):
    """
    centered, equal-sided kernel with side-length w
    uses any func : R^n -> R
    does not normalize
    """
    dom = np.indices(w)
    dom = dom - (w-1)/2
    res = [func(x) for x in dom.reshape(len(w),-1).T]
    res = np.array(res).reshape(dom.shape[1:])
    return res

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
