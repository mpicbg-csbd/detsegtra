from types import SimpleNamespace
from pathlib import Path,PosixPath
from skimage import io
import tifffile
import pickle
import json
import numpy as np
import os
import re
import torch
import collections
from scipy.io import loadmat

import ipdb

def clean(s):
  "replaces arbitrary string with valid python identifier (SimpleNamespace attributes follow same rules as python identifiers)"

  s = str(s)
  ## Remove invalid characters
  s = re.sub('[^0-9a-zA-Z_]', '', s)
  ## Remove leading characters until we find a letter or underscore
  s2 = re.sub('^[^a-zA-Z_]+', '', s)
  ## fix pure-numbers dirnames "01" → "d01"
  s = s2 if s2 else "d"+s
  return s

known_filetypes = ['.npy', '.png', '.tif', '.tiff', '.pkl', '.json', '.mat']
known_scalars = [bool,int,tuple,range,float,str,bytes,Path,PosixPath]
known_py_collections = [dict, set, list]
# known_array_collection = [np.ndarray, torch.Tensor]

def _is_scalar(x):
  if type(x) in known_scalars: return True
  if type(x) is np.ndarray and x.ndim==0: return True
  if type(x) is torch.Tensor and x.ndimension==0: return True
  return False

def _is_collection(x):
  if type(x) in known_py_collections: return True
  if type(x) is np.ndarray and x.ndim>0: return True
  if type(x) is torch.Tensor and x.ndimension()>0: return True
  return False

def save(d, base):
  
  base = Path(base).resolve()

  if base.suffix in extension_to_write.keys():
    base.parent.mkdir(parents=True,exist_ok=True)
    f = extension_to_write[base.suffix]
    f(base, d)
    return

  assert type(d) is SimpleNamespace

  scalars = SimpleNamespace()
  for k,v in d.__dict__.items():

    if _is_collection(v): _save_file(base,k,v)
    
    elif type(v) is SimpleNamespace:
      save(v,base/str(k))
    
    else:
      assert _is_scalar(v)
      scalars.__dict__[k] = v
      # print("Scalar key,val: ", k, v)

  pickle.dump(scalars,open(base/"scalars.pkl",'wb'))

def _save_file(dir,name,v):
  dir = Path(dir); dir.mkdir(parents=True,exist_ok=True)
  name = str(name)

  if type(v) is torch.Tensor: v = v.numpy()

  if type(v) is np.ndarray and v.dtype == np.uint8 and (v.ndim==2 or (v.ndim==3 and v.shape[2] in [3,4])):
    io.imsave(dir/(name +'.png'),v)
  elif type(v) is np.ndarray:
    file = str(dir/(name +".tif"))
    tifffile.imsave(file,v,compress=0)
    # np.save(dir/(name +'.npy'),v)
  elif type(v) in known_py_collections:
    try:
      json.dump(v,open(dir/(name +'.json'),'w'))
    except:
      os.remove(dir/(name +'.json'))
      pickle.dump(v,open(dir/(name +'.pkl'),'wb'))

def load(base,filtr='.'):
  res  = dict()
  base = Path(base).resolve()

  if base.is_file(): return _load_file(base) ## ignore filter

  from segtools.python_utils import sorted_alphanum
  for d in sorted_alphanum(map(str, (base.iterdir()))):
    print(d)
    d=Path(d)
    d2 = clean(d.stem)

    if d2 in res.keys():
      print("double assignment: ", str(d), d2)
      # if d.suffix == '.pkl':
      #   d2 = d.parent / d.stem / '.json'
      #   if 

    if d.is_dir():
      obj = load(d,filtr=filtr)
      if len(obj.__dict__)>0:
        res[d2] = obj

    if d.is_file() and d.suffix in known_filetypes and re.search(filtr,str(d)):

      obj = _load_file(d)

      if d.name=="scalars.pkl":
        for k,v in obj.__dict__.items():
          res[k] = v
      else:
        res[d2] = obj

  return SimpleNamespace(**res)

extension_to_read = {
  '.npy': lambda f : np.load(f),
  '.png': lambda f : np.array(io.imread(f)),
  '.tif': lambda f : tifffile.imread(str(f)),
  '.tiff':lambda f : tifffile.imread(str(f)),
  '.pkl': lambda f : pickle.load(open(f,'rb')),
  '.json':lambda f : json.load(open(f,'r')),
  '.mat': lambda f : SimpleNamespace(**loadmat(f)),
  }

extension_to_write = {
  '.npy':lambda  f,x : np.save(f,x),
  '.png':lambda  f,x : io.imsave(f,x),
  '.tif':lambda  f,x : tifffile.imsave(str(f),x),
  '.tiff':lambda f,x : tifffile.imsave(str(f),x),
  '.pkl':lambda  f,x : pickle.dump(x,open(f,'wb')),
  '.json':lambda f,x : json.dump(x,open(f,'w')),
  }


def _load_file(name):
  name = Path(name)
  f = extension_to_read[name.suffix]
  return f(name)
  # f = Path(f)
  # if f.suffix=='.npy':
  #   return np.load(f)
  # if f.suffix=='.png':
  #   return np.array(io.imread(f))
  # if f.suffix in ['.tif','.tiff']:
  #   return tifffile.imread(str(f))
  # if f.suffix=='.pkl':
  #   return pickle.load(open(f,'rb'))
  # if f.suffix=='.json':
  #   try:
  #     x=json.load(open(f,'r'))
  #   except Exception as e:
  #     print(e)
  #     x=[]
  #   return x

def toarray(sn):
  assert type(sn) is SimpleNamespace
  # def f(x): print(x);return x
  return np.array([x for x in sn.__dict__.values()]) # if type(x) is np.ndarray])

def flatten(l):
  for el in l:
    if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
      yield from flatten(el)
    else:
      yield el

def flatten_sn(l):
  return SimpleNamespace(**flatten_nested_dicts(l))

def flatten_nested_dicts(d, parent_key='', sep='_'):
  items = []
  if type(d) is SimpleNamespace: d = d.__dict__
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping) or isinstance(v,SimpleNamespace):
      items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)
