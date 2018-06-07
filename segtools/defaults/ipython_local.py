from .ipython import *

## visual stuff relying on anaconda
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
plt.switch_backend('qt5agg')
import seaborn as sns

## my local code

from .. import spima
from .. import track_tools
from .. import nhl_tools

sys.path.insert(0,'/Users/broaddus/Desktop/Projects/')
from stackview.stackview import Stack #, StackQt

## martin's stuff
import gputools

## martin's visual stuff
import spimagine

from subprocess import run

def qopene():
  res = run(['rsync efal:qsave.npy .'], shell=True)
  print(res)
  return np.load('qsave.npy')

def qopen():
  # import subprocess
  res = run(['rsync broaddus@falcon:qsave.npy .'], shell=True)
  print(res)
  return np.load('qsave.npy')

def sync(name, external=False):
  cmd = "rsync -rav --exclude='*.npy' --exclude='*.npz' --exclude='*.h5' falcon:/projects/project-broaddus/fisheye/{0}/* {0}".format(name)
  if external:
    cmd = "rsync -rav --exclude='*.npy' --exclude='*.npz' --exclude='*.h5' efal:{0}/* {0}".format(name)
  res = run([cmd], shell=True)
  print(res)

def update_stack(iss, img, hyp, r, nhl):
  img2 = img.copy()
  mask = nhl_tools.mask_nhl(nhl, hyp)
  img2[mask] = img2[mask]*r
  iss.stack = img2

newcents = []
def onclick_centerpoints(event):
  xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
  zi = iss.idx[0]
  print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
      (event.button, event.x, event.y, event.xdata, event.ydata))
  print(zi, yi, xi)
  if event.key=='C':
    print('added! ', event.key)
    newcents.append([zi,yi,xi])
# cid = iss.fig.canvas.mpl_connect('button_press_event', onclick_centerpoints)





