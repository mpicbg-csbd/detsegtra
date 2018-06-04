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
# from subprocess import run
from glob import glob
from collections import Counter
from math import ceil,floor

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
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import watershed
import skimage.io as io

## my own stuff
from .. import color
from .. import nhl_tools
from .. import scores_dense as ss
from .. import patchmaker as patch
from .. import plotting
from ..numpy_utils import *
from ..python_utils import *



def qsave(x):
  np.save('qsave', x)

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

