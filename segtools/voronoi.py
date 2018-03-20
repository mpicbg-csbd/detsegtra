import numpy as np

import scipy.spatial as spatial
from pykdtree.kdtree import KDTree as pyKDTree
from skimage import measure
import scipy.ndimage as nd
from numba import jit


def voronoi3d(coords, dmax=20):
    # dlny = spatial.Delaunay(coords[:,[0,1]])
    # spatial.delaunay_plot_2d(dlny)

    # dlny = spatial.Delaunay(coords)
    voro = spatial.Voronoi(coords)
    distances = [np.linalg.norm(coords[x]-coords[y]) for x,y in voro.ridge_points]
    distances = np.array(distances)
    # distances.max()
    # plt.hist(distances, bins=200)

    # spimagine.volshow(img)
    # img.shape
    # img.max(axis=0)

    edges, distances_filtered = [], []
    for (i,j), d in zip(voro.ridge_points, distances):
        if d < dmax:
            edges.append((i,j))
            distances_filtered.append(d)
    edges = np.array(edges)
    distances_filtered = np.array(distances_filtered)

    rmax = edges.max()

    mm = [[] for _ in range(rmax+1)]

    for i in range(len(edges)):
        v1, v2 = edges[i]
        d = distances_filtered[i]
        mm[v1].append(v2)
        mm[v2].append(v1)
    histy = [len(m) for m in mm]
    # plt.hist(histy, bins=100)
    return voro, distances, histy

def voronoi_kd(coords, imshape, maxdist=20):
    ndim = len(imshape)
    kdt = pyKDTree(coords)
    idximg = np.indices(imshape)
    dist, idxlist = kdt.query(idximg.reshape(ndim, -1).T, k=1)
    distimg = dist.reshape(imshape)
    idximg = idxlist.reshape(imshape)
    idximg += 1
    if maxdist:
        mask = distimg > maxdist
        idximg[mask] = 0
    return idximg, distimg

@Incomplete
def identify_neibs(lab):
    """
    TODO: should be able to go back and forth between binary structure and list of pixel neighbors
    """
    bs = nd.generate_binary_structure(3,1)
    neibs = np.indices(bs.shape)
    l = [v-[1,1,1] for v in neibs.reshape(3,-1).T if bs[v[0],v[1],v[2]]==1]
    l = np.array(l)

def label_boundaries_from_direction3d(lab, vec):
    a,b,c = vec
    assert min(a,b,c) >= 0
    v1 = lab[a:,b:,c:]
    def neg_slice(x):
        if x==0:
            return None
        else:
            return -x
    a,b,c = map(neg_slice, [a,b,c])
    v2 = lab[:a,:b,:c]
    print(v1.shape, v2.shape)
    res = v1 != v2
    return np.stack([v1[res], v2[res]])

def label_boundaries_from_direction2d(lab, vec):
    a,b = vec
    assert min(a,b) >= 0
    v1 = lab[a:,b:]
    def neg_slice(x):
        if x==0:
            return None
        else:
            return -x
    a,b = map(neg_slice, [a,b])
    v2 = lab[:a,:b]
    print(v1.shape, v2.shape)
    res = v1 != v2
    return np.stack([v1[res], v2[res]])

def label_neighbors(lab, ndim=2):
    ls = 2*np.identity(ndim, dtype='int')
    if ndim==3:
        res = np.concatenate([label_boundaries_from_direction3d(lab, l) for l in ls], axis=1)
    elif ndim==2:
        res = np.concatenate([label_boundaries_from_direction2d(lab, l) for l in ls], axis=1)
    # now build a histogram over tuples? count edges ∀ pairs
    hist = dict()
    for l1,l2 in res.T:
        count = dict.get(hist, (l1,l2), 0)
        hist[(l1,l2)] = count+1
    return hist

def lab2binary_neibs(lab):
    res = lab.copy().astype(np.int)
    m1  = lab[1:, :] == lab[:-1, :]
    m2  = lab[:, 1:] == lab[:, :-1]
    res[:-1] = m1
    res[1:]  += m1
    res[:,1:] += m2
    res[:,:-1] += m2
    if lab.ndim==3:
        m3  = lab[:, :, 1:] == lab[:, :, :-1]
        res[:,:,1:] += m3
        res[:,:,:-1] += m3
    return res

def hist2neibs(hist, pixmax=5):
    neibs = dict()
    for l1, l2 in hist.keys():
        c = hist[(l1, l2)]
        if l1>0 and l2>0:
            if c > pixmax:
                neibs[l1] = neibs.get(l1, []) + [(l2, c)]
    # neibs = [list(g) for k,g in itertools.groupby(sorted(hist.keys()), lambda t: t[0])]
    # counts = [len(l) for l in neibs]
    return neibs

def xyz2rthetaphi(v):
    x,y,z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arctan(y/x)
    ph = np.arccos(z/r)
    return r,th,ph

def coords_xy_2_voronoi_theta_phi(coords):
    """
    polysides maps an nuclei id to the number of voronoi neighbors it has in the 2d r,theta,phi projection
    """
    cmean = coords.mean(axis=0)
    rthph_coords = np.array(list(map(xyz2rthetaphi, coords-cmean)))
    rthph_coords[:,2] = rthph_coords[:,2] * np.cos(rthph_coords[:,1])
    voro_thph = spatial.Voronoi(rthph_coords[:,[1,2]])
    polysides = [len(voro_thph.regions[voro_thph.point_region[p]]) for p in range(len(voro_thph.points))]
    return voro_thph, polysides

def disclination_charges(polysides):
    sum = 0
    for p in polysides:
        sum += 6-p
    return sum

@jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

# @jit("i4[:](i4[:,:,:], i8[:,:])", nopython=True)
def neib_me(lab, neiblist):
    counter = np.zeros(lab.max()+1, dtype='uint8')
    neibarrayset = np.zeros((lab.max()+1, 20))-1
    print(neibarrayset.shape)
    xx,yy,zz = lab.shape
    for i in range(1, xx-1):
        print('new z')
        for j in range(1, yy-1):
            for k in range(1, zz-1):
                homeval = lab[i,j,k]
                for a,b,c in neiblist:
                    awayval = lab[i+a,j+b,k+c]
                    idx = find_first(awayval, neibarrayset[homeval])
                    if idx == -1:
                        c = counter[homeval]
                        neibarrayset[c] = awayval
                        counter[homeval] += 1
    return neibarrayset

