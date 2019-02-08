import numpy as np

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import LassoSelector

def centerpoint_listener(iss, pointlist):
  def onclick_centerpoints(event):
    xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
    zi = iss.idx[0]
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    print(zi, yi, xi)
    if event.key=='c':
      print('added! ', event.key)
      pointlist.append([zi,yi,xi])
  cid = iss.fig.canvas.mpl_connect('button_press_event', onclick_centerpoints)
  return cid
  # return onclick_centerpoints

def centerpoints2slices(img, pointlist, dx=128):
  slices = []
  for pt in pointlist:
    img2 = img[pt[0]]
    img2 = np.pad(img2, dx, mode='reflect')
    sl = slice(pt[1], pt[1]+2*dx), slice(pt[2], pt[2]+2*dx)
    slices.append(img2[sl])
  return np.array(slices)

def lasso_randomPolygons(ax, polylist):
  def onselect(verts):
    p = Polygon(verts, fc=(random(), random(), random(), 0.25), rasterized=True)
    ax.add_patch(p)
    polylist.append(p)
    # mask = nxutils.points_inside_poly
  lasso = LassoSelector(ax, onselect)
  return lasso

def lasso_draw(ax, mask):
  def onselect(verts):
    # p = Polygon(verts, fc=(random(), random(), random(), 0.25), rasterized=True)
    # ax.add_patch(p)
    verts = np.array(verts, dtype=np.int)
    mask[verts[:,0], verts[:,1]] = mask.max() + 1
    # mask = nxutils.points_inside_poly
  lasso = LassoSelector(ax, onselect)
  return lasso

