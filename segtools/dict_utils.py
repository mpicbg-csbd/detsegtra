import numpy as np
from types import SimpleNamespace
## utils

def deepget(obj,indlist):
  d = obj
  for k in indlist:
    d = d[k]
  return d

def deepset(obj,indlist,value):
  d = obj
  for k in indlist[:-1]:
    d = d[k]
  d[indlist[-1]] = value

def invertdict(lod, f=lambda x: np.array(x)):
  d2 = dict()
  for k in lod[0].keys():
    d2[k] = f([x[k] for x in lod])
  return d2

def revertdict(dol):
  res = []
  for i in range(len(list(dol.values())[0])):
    print(i)
    res.append({k:v[i] for k,v in dol.items()})
  return res


def test_cartesian_indexable():
  dol = SimpleNamespace(a=np.random.rand(100),b=np.random.rand(100),c=np.array([x for x in "abcdefghij"])[np.random.randint(0,10,100)])







def updatekeys(d1,d2,keys):
  for k in keys:
    d1[k] = d2[k]
  return d1

def fromlocals(_locals,keys):
  d = dict()
  for k in keys:
    d[k] = _locals[k]
  return d

def selkeys(d,keys): return {k: d[k] for k in keys}


info = """

Sun Feb 23 17:38:57 2020

What would be a generic version of invert/revert dict?
This function is used whenever we have a list of dictionaries (w same keys) that we'd like to have as a dict of lists.
revert takes the dict of lists back to a list of dicts.
Things we might do:
- make this work for SimpleNamespace objects as well as dicts
- make it work for more than just lists...
- make it work for more deeply nested objects (lists of lists of dicts)

The fully generic idea is that of the transpose of an object which is indexed by some product space.
The input spaces may be ordered (e.g. integers index a list) or not (strings|attributes index a dict|namespace).

Implementations...
One idea is, instead of worrying about actually using dictionaries, etc, we convert everything to ndarry under the hood and keep track of a mapping from attributes/strings to integer indices.
We could also use this to turn sets of keys into subsets of int dimensions, etc.
What properties should this object have? In particular, do we need to enforce type consistency along at least _one_ axis? Along every integer-indexable axis?
Type consistency along every int axis would mean that specifying all the N-m not-int axes (on indexable with ndim=N) would be enough to tell us the type, and would return a type-homogeneous ndarray with ndim=m.
But changing the value of any non-int axis could change the type of the ndarray.
How does this idea relate to a DataTable/DataFrame in Pandas/R/Julia/etc?
I think a DataFrame is supposed to mirror a spreadsheet. It only has two axes. One is int-indexable and type homogeneous. These are called "columns".
The other has arbitrary, unsorted indices and is type inhomogeneous. These are called "rows".
Tables with more than two dimensions are partially supported and usually very awkward to work with.
It is usually possible to flatten your sorted/unsorted dimensions so that your data fits this 2D model.



"""