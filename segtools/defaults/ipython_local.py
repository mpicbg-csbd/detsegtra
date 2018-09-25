from .ipython import *

## visual stuff relying on anaconda
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
plt.switch_backend('qt5agg')

import seaborn as sns
from subprocess import run

## my local code

from ..cell_view_lib import imshowme
from .. import spima
from .. import track_vis

sys.path.insert(0,'/Users/broaddus/Desktop/Projects/')
from stackview.stackview import Stack #, StackQt

## martin's stuff
import gputools

## martin's visual stuff
import spimagine

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
  cmd = "rsync -rav --exclude='*.npy' --exclude='*.tif' --exclude='*.npz' --exclude='*.h5' --exclude='*.pkl' "
  if external:
    cmd += "efal:{0}/* {0}".format(name)
  else:
    cmd += "falcon:/projects/project-broaddus/fisheye/{0}/* {0}".format(name)

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
  print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
  print(zi, yi, xi)
  if event.key=='C':
    print('added! ', event.key)
    newcents.append([zi,yi,xi])

# cid = iss.fig.canvas.mpl_connect('button_press_event', onclick_centerpoints)

def open_in_preview(*args, normed=True, rm_old=True):
  "save all arrays and open in preview"
  def norm(img): img = (img-img.min())/(img.max() - img.min()); return img;
  dir0 = Path('imshow')
  dir0.mkdir(exist_ok=True)
  if rm_old:
    for d in dir0.iterdir():
      os.remove(d)
  def a():
    g = glob('imshow/*.png')
    if len(g)==0: return 0
    s = sorted(g)[-1][-7:-4]
    n = int(s)
    print(g,s,n)
    return n
  number = a()
  print(number)
  for i,arr in enumerate(args):
    # if 'float' in str(arr.dtype):
    if normed==True:
      arr = norm(arr)
    io.imsave(dir0 / 'img{:03d}.png'.format(i + number + 1), arr)
  cmd = "open -a preview imshow"
  res = run([cmd], shell=True)  




