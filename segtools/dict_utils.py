import numpy as np
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