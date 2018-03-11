# make color remapping dict
# get all the children of the branching roots
# for each root, get id, get children, get children id, map children id to id, (ignore time in id! make list of dicts to represent time.)

import numpy as np
from numba import jit
from tabulate import tabulate
import skimage.io as io


def make_combo_img(img, den, hyp):
    """
    Combine the three image types and normalize so they can be
    viewed in spimagine without having to adjust contrast.
    """
    modimg = mod_color_hypimg(hyp)
    modimg = (modimg*(img.max()/modimg.max()))
    den *= img.max()/den.max()
    print(img.min(), img.max())
    print(den.min(), den.max())
    print(modimg.min(), modimg.max())
    img = img.astype('float32')*1.4
    res = np.stack((img,den,modimg), axis=0)
    return res

def mod_color_hypimg(hyp):
    """
    give a hypothesis image
    return an image with values meant for spimagine display
    when plotting labels. Remember to keep a mask of the zero-values before you mod. Then reset zeros to zero after adding 2x the mod value.
    """
    hyp2 = hyp.copy()
    mask = hyp2==0
    hyp2 %= 7
    hyp2 += 5
    hyp2[mask] = 0
    return hyp2

def update_w(w, img):
    w.glWidget.renderer.update_data(img)
    w.glWidget.refresh()

def get_cube_from_transform(img, tcube):
    """
    specify a spimagine.TransformData and full-sized image
    get the image cube inside the bounding box.
    """
    zhw,yhw,xhw = np.array(img.shape)/2
    tcube.bounds 
    xmin = int((1 + tcube.bounds[0])*xhw)
    xmax = int((1 + tcube.bounds[1])*xhw)
    ymin = int((1 + tcube.bounds[2])*yhw)
    ymax = int((1 + tcube.bounds[3])*yhw)
    zmin = int((1 + tcube.bounds[4])*zhw)
    zmax = int((1 + tcube.bounds[5])*zhw)
    cube = img[zmin:zmax, ymin:ymax, xmin:xmax]
    return cube

def get_slices_from_transform(img, tcube):
    """
    specify a spimagine.TransformData and full-sized image
    get the image cube inside the bounding box.
    """
    zhw,yhw,xhw = np.array(img.shape)/2
    tcube.bounds
    xmin = int((1 + tcube.bounds[0])*xhw)
    xmax = int((1 + tcube.bounds[1])*xhw)
    ymin = int((1 + tcube.bounds[2])*yhw)
    ymax = int((1 + tcube.bounds[3])*yhw)
    zmin = int((1 + tcube.bounds[4])*zhw)
    zmax = int((1 + tcube.bounds[5])*zhw)
    slt = slice(zmin,zmax), slice(ymin,ymax), slice(xmin,xmax)
    return slt

# ---- recolor images according to lineage

def recolor_by_lineage(hypimgs, tv, te):
    cmap = dr.build_colormap(te, tv)
    cmap_arr = list(map(cmap_dict_to_array, cmap))
    hypimgs_recolored = [permute_array_numba(img, cmap) for (img,cmap) in zip(hypimgs, cmap_arr)]
    hypimgs_recolored = np.array(hypimgs_recolored)
    return hypimgs_recolored

def highlight_something_across_many_frames(denimgs, hypimgs, tv, tb):
    """
    denimgs, hypimgs are lists of ndarray images.
    tv,tb are truevertices and truebranching.
    """
    reslist = []
    # something = make_about_to_divide_list
    something = make_about_to_disappear_list
    start = 0
    end = len(denimgs)-1
    for t in range(start, end):
        divlist = something(t, hypimgs, tv, tb)
        reslist.append(highlight_labels(denimgs[t], hypimgs[t], divlist))
    reslist = np.array(reslist)
    return reslist

def make_about_to_disappear_list(time, hypimgs, tv, tb):
    disappearlist = np.ones(hypimgs[time].max()+1, dtype=np.uint16)
    for n in tv[time]:
        if len(tb[n]) == 0:
            print(n)
            ind = int(n[1:n.index('t')])
            disappearlist[ind] = 2
    return disappearlist

def make_about_to_divide_list(time, hypimgs, tv, tb):
    divisionlist = np.ones(hypimgs[time].max()+1, dtype=np.uint16)
    for n in tv[time]:
        if len(tb[n]) == 2:
            print(n)
            ind = int(n[1:n.index('t')])
            divisionlist[ind] = 2
    return divisionlist

def make_just_divided_list(time, hypimgs, tv, tb):
    divisionlist = np.ones(hypimgs[time].max()+1, dtype=np.uint16)
    for n in tv[time-1]:
        if len(tb[n]) == 2:
            for n2 in list(tb[n].keys()):
                print(n2)
                ind = int(n2[1:n2.index('t')])
                divisionlist[ind] = 2
    return divisionlist

def highlight_replace(hyp, scale):
    print(hyp.dtype, hyp.min(), hyp.max())
    print(scale.dtype, scale.min(), scale.max())

    uint_ts  = [np.uint8, np.uint16, np.uint32, np.uint64]
    int_ts   = [np.int8, np.int16, np.int32, np.int64]
    float_ts = [np.float16, np.float32, np.float64, np.float128]
    
    assert hyp.dtype in uint_ts + int_ts
    hyp = hyp.astype(np.uint64)

    assert hyp.ndim == 3
    assert scale.ndim == 1 # TODO: dispatch on ndim for RGB[A] recoloring
    assert hyp.min() == 0
    assert hyp.max()+1 == len(scale)

    scaletype = scale.dtype
    if scaletype in uint_ts:
        scale = scale.astype(np.uint64)
        res = numba_replace_uint(hyp, scale)
    elif scaletype in int_ts:
        scale = scale.astype(np.int64)
        res = numba_replace_int(hyp, scale)
    elif scaletype in float_ts:
        scale = scale.astype(np.float64)
        res = numba_replace_float(hyp, scale)

    return res

@jit('u8[:,:,:](u8[:,:,:], u8[:])')
def numba_replace_uint(arr, ind_arr):
    res = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                res[i,j,k] = ind_arr[arr[i,j,k]]
    return res

@jit('i8[:,:,:](u8[:,:,:], i8[:])')
def numba_replace_int(arr, ind_arr):
    res = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                res[i,j,k] = ind_arr[arr[i,j,k]]
    return res

@jit('f8[:,:,:](u8[:,:,:], f8[:])')
def numba_replace_float(arr, ind_arr):
    res = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                res[i,j,k] = ind_arr[arr[i,j,k]]
    return res

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # import numpy as np
    from scipy import linalg as LA
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
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

def mk_quat(m):
    "this is the boundary from our data (z,y,x) indicies. To spimagine data: (x,y,z) inds."
    m = m[:, [2,1,0]]
    w = np.sqrt(1.0 + m[0,0] + m[1,1] + m[2,2]) / 2.0;
    w4 = (4.0 * w);
    x = (m[2,1] - m[1,2]) / w4
    y = (m[0,2] - m[2,0]) / w4
    z = (m[1,0] - m[0,1]) / w4
    # TODO: Turn this into a quaternion OUTSIDE the drawing namespace!
    return (w,x,y,z)

def hypothesis_img(nuc, hyp, img, width=40):
    """
    Takes a nuclear hypothesis, a HypothesisImage and an IntensityImage and returns
    a cube around that hypothesis where it's highlighted (two cubes. one intensity based and one label based)
    """
    z,x,y = list(map(int, nuc['coords']))
    r = width
    # from matplotlib.mlab import PCA
    # zm,xm,ym = hyp.shape
    # zstart,zend = max(0, z-r, key)
    labelcube = hyp[z-r:z+r, x-r:x+r, y-r:y+r].copy()
    print((labelcube.shape))
    intenscube = img[z-r:z+r, x-r:x+r, y-r:y+r].copy()
    print(("Uniques: ", np.unique(labelcube)))
    # this is not the correct time
    # nhl = graph_builder2.image_to_nuclear_hypothesis_list(labelcube, intenscube, 0)
    # centers = np.array([n['coords'] for n in nhl])
    # dat,val,m = PCA(centers, dims_rescaled_data=3)
    the_label = int(nuc['label'])
    # mask = labelcube==the_label
    mask = labelcube != 0
    ind = np.indices(labelcube.shape)
    m = None
    pca_orient = False
    if pca_orient:
        dat,val,m = PCA(ind[:, mask].T.astype('float32'), dims_rescaled_data=3)
        print("Vectors are: ", m)
        print("Eigvalues are: ", val)
    print(("Label: ", nuc['label']))
    labelmask = (labelcube != nuc['label']) & (labelcube != 0)
    intensmask = labelcube == the_label
    print(("Mask sum: ", intensmask.sum()))
    # FIXME: this will break when the labels are 0,1 and 2.
    labelcube[labelmask] = int(the_label/2.0)
    intenscube[intensmask] = (1.5 * intenscube[intensmask]) #.astype('uint16')
    # assert False
    # print("Uniques 2: ", np.unique(labelcube))
    # print(np.sum(labelmask))
    return intenscube, labelcube, m

def sorted_uniques(img):
    a,b = np.unique(img, return_counts=True)
    counts = sorted(zip(a,b), key=lambda c: c[1])
    return counts

def build_size_colormap(list_of_nhds, tv=None):
    """
    takes TrueEdges and TrueVertices and HypothesisGraph.
    returns a list of colormaps by size.
    """
    list_of_colormaps = []
    for i in range(len(list_of_nhds)):
        d = dict()
        nhd = list_of_nhds[i]
        for v,feats in list(nhd.items()):
            nid = feats['label']
            d[nid] = feats['area']
        list_of_colormaps.append(d)
    return list_of_colormaps

def build_colormap(te, tv):
    """
    takes a TrueBranching and TrueVertices. 
    returns a list of colormaps where the label is passed down to children.
    spontaneous appearances are given a new, unique id.
    """
    list_of_colormaps = []
    true_ids = [[int(x[1:x.index('t')]) for x in vertlist] for vertlist in tv]

    # TODO: we don't know all the possible id's from the original image. We have to either keep them in the graph or get the image or something...
    current_colormap = dict()
    for nuclei in true_ids[0]:
        current_colormap[nuclei] = nuclei
    list_of_colormaps.append(current_colormap)

    for i in range(len(tv[:-1])):
        current_nuclei = tv[i]
        next_nuclei = tv[i+1]
        edges = te[i]
        current_colormap = _color_next_timepoint(edges, current_nuclei, next_nuclei, current_colormap)
        list_of_colormaps.append(current_colormap)
    # assert every id is mapped to!
    assert list(map(len, list_of_colormaps)) == list(map(len, tv))
    return list_of_colormaps

def _color_next_timepoint(edges, current_nuclei, next_nuclei, current_colormap):
    "Used by build_colormap as the step function."
    next_colormap = dict()

    mapping = dict()
    for (n1,n2) in edges:
        mapping[n2] = n1

    maxid = max(current_colormap.values())
    for n2 in next_nuclei:
        n2id = int(n2[1:n2.index('t')])
        try:
            n1 = mapping[n2]
            n1id = int(n1[1:n1.index('t')])
            next_colormap[n2id] = current_colormap[n1id]
        except: # we have an exception iff mapping is missing n2 i.e. n2 is not in an (n1,n2) edge. n2 is an appearance.
            # next_colormap[n2id] = 9999 # means "missing parent"
            next_colormap[n2id] = maxid
            maxid += 1
        
    next_nuclei = set(next_nuclei)
    print(next_nuclei - set(mapping.keys()))
    return next_colormap

def cmap_dict_to_array(cmap):
    "converts colormap dictionaries to uint16 ndarrays for use in permute_array_numba"
    arr = np.zeros(max(cmap.keys())+1).astype('uint16')
    for k,v in list(cmap.items()):
        arr[k] = v
    return arr



# @jit # using the version with types included gives 10x speedup (400ms)
@jit('i8[:,:,:,:](i8[:,:,:],i8[:,:])')
def permute_array_numba_RGB(arr, ind_list):
    """
    replaces  arr[i,j,k] with ind_list[arr[i,j,k]]
    use this instead of recolor_image
    """
    res = np.zeros(arr.shape + (3,))

    if arr.ndim==3:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    res[i,j,k,0] = ind_list[arr[i,j,k],0]
                    res[i,j,k,1] = ind_list[arr[i,j,k],1]
                    res[i,j,k,2] = ind_list[arr[i,j,k],2]
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                res[i,j,0] = ind_list[arr[i,j],0]
                res[i,j,1] = ind_list[arr[i,j],1]
                res[i,j,2] = ind_list[arr[i,j],2]
    return res


# def label_to_RGB(label, N=10, sat=0.275, bri=0.75, bg='black'):
@DeprecationWarning
def hyp2rgb_random(label, N=10, sat=0.275, bri=0.75, bg='black'):
    """
    takes a hypothesis image and colors the labels with RGB.
    """
    import skimage
    import colorsys

    # HSV_tuples = [(x * 1.0 / N, sat, bri) for x in range(N)]
    # RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    import seaborn as sns
    RGB_tuples = sns.color_palette('hls', N)
    if bg=='black':
        RGB_tuples.insert(0, (0,0,0))
    elif bg=='white':
        RGB_tuples.insert(0, (1,1,1))

    RGB_tuples2 = np.array(RGB_tuples)

    print("label: ", label.max(), label.min())

    mask = label==0
    label %= N
    label += 1
    label[mask] = 0
    res = permute_array_numba_RGB(label, RGB_tuples2)

    # label2rgb doesn't work with 3d images! yes it does???
    # res = skimage.color.label2rgb(label, colors=RGB_tuples, bg_label=0, bg_color=(0, 0, 0))
    # print fancy.dtype, fancy.max(), fancy.min()
    return res

def merge_label_intens(label, intens):
    """
    Used for building SumImage for RGB movies of tracking.
    label and intens are single timepoint ndarrays.
    """
    intens *= 2**16/intens.max()
    intens = intens.astype('uint16')
    print("intens: ", intens.dtype, intens.max(), intens.min())

    fancy = hyp2rgb_random(label)
    intens = skimage.color.gray2rgb(intens)
    intens_norm = intens.astype('float64') / intens.max()
    res = fancy + 0.5*intens_norm
    res /= res.max()
    res = (2**16 * res).astype('uint16')
    return res


# ------------------- DEPRECATED ------------------------------------------------

@DeprecationWarning
def recolor_image(img, colormap):
    """
    img is an ndarray. Only use single timepoints, as that's where labels are unique after segmentation.
    colormap is a dict|list|array that maps from old labels to new ones
    """
    c = img.copy()
    it = np.nditer(c, flags=['multi_index'])
    colormap[0] = 0
    count = 0
    while not it.finished:
        try:
            c[it.multi_index] = colormap[img[it.multi_index]]
        except KeyError as e:
            print(("key error w value ", img[it.multi_index]))
            c[it.multi_index] = 0
        it.iternext()
        if count % 10000 == 0:
          print((it.multi_index))
        count += 1
    return c
    # for k,v in colormap.iteritems():
    #     c[img==k] = v
    # return c

@DeprecationWarning
def permute_labels(lab, perm):
    """TODO: UNFINISHED"""
    img3 = np.zeros(lab.shape)
    it = np.nditer(img3, flags=['multi_index', 'write_only'])
    while not it.finished:
        img3[it.multi_index] = perm[lab[it.multi_index]+1]
        it.iternext()
        if count % 10000 == 0:
          print((it.multi_index))
        count += 1

# from skimage.transform import  resize
# a,b,c = res.shape[:3]
# print "abc = ", a, b, c
# from scipy.interpolate import interpnd
# from scipy.ndimage import zoom
# zoom()
# res = resize(res, (3*a, 3*b, 3*c), order=3)