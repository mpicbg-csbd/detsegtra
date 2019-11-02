from types import SimpleNamespace
from pathlib import Path
from skimage import io
import tifffile
import pickle
import json
import numpy as np
import os

readable_filetypes = ['.npy', '.png', '.tif', '.pkl', '.json',]
saveable_pytypes = [np.ndarray, dict, set, list]

def ns2dir(d, dir):
  dir = Path(dir); dir.mkdir(parents=True,exist_ok=True)
  
  for k,v in d.__dict__.items():
    print(k,v,type(v))
    if type(v) in saveable_pytypes:
      save(dir,k,v)
    elif type(v) is SimpleNamespace:
      ns2dir(v,dir/str(k))
    else:
      print("Type not saveable: ", k, v)

def save(dir,k,v):
  if type(v) is np.ndarray:
    if v.dtype == np.uint8:
      io.imsave(dir/(str(k)+'.png'),v)
    # elif 'float' in str(v.dtype) and v.max()<=1.0:
    #   io.imsave(dir/(str(k)+'.png'),v)
    else:
      file = str(dir/(str(k)+'.tif'))
      tifffile.imsave(file,v,compress=0)
      # np.save(dir/(str(k)+'.npy',v))
  elif type(v) in [dict,set,list]:
    json.dump(v,open(dir/(str(k)+'.json'),'w'))
    # pickle.dump(v,open(dir/(str(k)+'.pkl'),'wb'))

def dir2ns(dir):
  res = dict()
  for root,dirs,files in os.walk(dir):
    root = Path(root)
    for d in dirs:
      res[d] = dir2ns(root/d)
    for f in files:
      f = Path(f)
      if f.suffix in readable_filetypes:
        res[f.stem] = load(root/f)
  return SimpleNamespace(**res)

def load(f):
  f = Path(f)
  if f.suffix=='.npy':
    return np.load(f)
  if f.suffix=='.png':
    return np.array(io.imread(f))
  if f.suffix=='.tif':
    return tifffile.imread(str(f))
  if f.suffix=='.pkl':
    return pickle.load(open(f,'rb'))
  if f.suffix=='.json':
    return json.load(open(f,'r'))
