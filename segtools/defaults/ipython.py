## python defaults
import sys
import os
import shutil
import json
import pickle
import random
import re
import itertools
from time import time
import subprocess
from glob import glob

## python 3 only
from pathlib import Path
from functools import reduce

## stuff I've had to install
from tabulate import tabulate

## anaconda defaults
# import networkx as nx
# import pandas as pd
import numpy as np
from tifffile import imread, imsave
from scipy.ndimage import zoom, label, distance_transform_edt, rotate
from scipy.ndimage.filters import convolve
from scipy.signal import gaussian, fftconvolve
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.io as io

## my own stuff
from .. import color
from .. import nhl_tools
# from .. import track_tools
from .. import scores_dense
from .. import patchmaker
from .. import plotting
from .. import math_utils
from .. import augmentation
from .. import stack_segmentation

from ..numpy_utils import *
from ..python_utils import *


def moviesave(arr,name='out.mp4',rate=4,rewrite=True):
  "axes are TYX[C]"
  Path('movie').mkdir(exist_ok=True)
  for i,x in enumerate(arr):
    if os.path.exists('movie/res{:03d}.png'.format(i)) and rewrite is False: pass
    io.imsave('movie/res{:03d}.png'.format(i),x)
  cmd = 'ffmpeg -y -r {rate} -i "movie/res%03d.png" -vf "fps=25,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2" {name}'.format(rate=rate,name=name)
  subprocess.run([cmd], shell=True)

def qsave(x):
  np.save('qsave', x)

def pload():
  x = pickle.load(open('psave.pkl','rb'))
  return x

def psave(x):
  pickle.dump(x,open('psave.pkl','wb'))

def ensure_exists(dir):
  try:
    os.makedirs(dir)
  except FileExistsError as e:
    print(e)

def run_from_ipython():
  "https://stackoverflow.com/questions/5376837/how-can-i-do-an-if-run-from-ipython-test-in-python"
  try:
      __IPYTHON__
      return True
  except NameError:
      return False

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif type(obj) in [np.float16, np.float32, np.float64, np.float128]:
      return float(obj)
    elif type(obj) in [np.int8, np.int16, np.int32, np.int64]:
      return int(obj)
    elif type(obj) in [np.uint8, np.uint16, np.uint32, np.uint64]:
      return int(obj)
    return json.JSONEncoder.default(self, obj)

def add_numbered_directory(path, base):
  s = re.compile(base + r"(\d{3})")
  def f(dir):
    m = s.search(dir)
    return int(m.groups()[0])
  drs = [f(d) for d in os.listdir(path) if s.search(d)]
  new_number = 0 if len(drs)==0 else max(drs) + 1
  newpath = str(path) + '/' + base + '{:03d}'.format(new_number)
  newpath = Path(newpath)
  newpath.mkdir(exist_ok=False)
  return newpath

