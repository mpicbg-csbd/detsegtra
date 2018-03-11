# loc_utils
import os
from functools import wraps
from time import time
import numpy as np
import re
from tabulate import tabulate

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap

def hist_sparse(arr, bins=500):
    """
    We should extend the histogram function s.t. it always does this, and we don't have shitty plots. Also, if we give bins=-1, 
    we should just get back np.unique(arr, return_counts=True)...
    """
    arr = np.array(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    if bins==-1:
        vals, counts = np.unique(arr, return_counts=True)
        dx = 0
    else:
        hist_pimg = np.histogram(arr, bins=bins)
        counts, vals, dx = hist_pimg[0], (hist_pimg[1][1:] + hist_pimg[1][:-1])/2.0, hist_pimg[1][1]-hist_pimg[1][0]
        m = counts!=0
        counts = counts[m]
        vals   = vals[m]
    return counts, vals, dx

def hist_dense(arr, bins=500):
    """
    We should extend the histogram function s.t. it always does this, and we don't have shitty plots. Also, if we give bins=-1, 
    we should just get back np.unique(arr, return_counts=True)...
    """
    arr = np.array(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    hist_pimg = np.histogram(arr, bins=bins)
    counts, vals, dx = hist_pimg[0], (hist_pimg[1][1:] + hist_pimg[1][:-1])/2.0, hist_pimg[1][1]-hist_pimg[1][0]
    return counts, vals, dx

def path_base_ext(fname):
    directory, base = os.path.split(fname)
    base, ext = os.path.splitext(base)
    return directory, base, ext

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect.
        taken from https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def pprint_list_of_dict(lod, keys=None):
  if not keys:
    keys = lod[0].keys()
  table = [keys]
  table = table + [d.values() for d in lod]
  print(tabulate(table))

def list2dist(lst):
    val, cts = np.unique(lst, return_counts=True)
    dist = dict(zip(val, cts))
    return dist
