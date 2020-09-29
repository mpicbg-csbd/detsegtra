"""
This whole script is deprecated.
Better to make movies using the imageio interface.
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
plt.ion()
# plt.switch_backend('qt5agg')

def imshow2(img, dpi_ratio=1, interpolation='nearest', cmap='gray', **kwargs):
  """
  `fig1 = plt.figure()` must be called separately otherwise we get "No renderer defined" error if
  we use the `fig1.draw` method. The workaround is to call fit_fig_to_axis TWICE!
  This is the only way to both *create* a figure, and resize it in a single call.
  """
  fig = plt.figure()
  ax = fig.gca()
  ax.imshow(img, interpolation=interpolation, cmap=cmap, **kwargs)
  ax.axis('off')
  ax.set_position([0,0,1,1])
  w,h = fit_fig_to_axis(fig)
  w,h = fit_fig_to_axis(fig)
  dpi = int(dpi_ratio*img.shape[0]/h)
  fig.set_dpi(dpi)
  return fig,ax

def fit_fig_to_axis(fig,scale=1):
  bbox = fig.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.set_size_inches(bbox.width*scale, bbox.height*scale, forward=True)
  # fig.set_size_inches(bbox.width*scale, bbox.height*scale, forward=True)
  return bbox.width, bbox.height

!ffmpeg -i "res000.avi" -f image2 "frames/img%03d.png"
!mkdir frameout

for i in range(1,12):
  img = io.imread('frames/img{:03d}.png'.format(i))
  fig,ax = imshow2(img.transpose((1,0,2))[::-1])
  ax.text(20,100,"{:d} mins".format(5*(i-1)),color='w',fontsize=20)
  fig.savefig('frameout/img{:03d}.png'.format(i),dpi=fig.dpi)
  ## dpi='figure' doesn't work :( and default is shit.

!ffmpeg -y -r 4 -i "frameout/img%03d.png" -vf "fps=25,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2" res000_2.mp4