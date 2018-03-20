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

## recolor images according to lineage

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
