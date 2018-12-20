## Additional Matplotlib functionality
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
plt.ion()

import sys

# ---- interactive 2d + 3d viewing ----

def imshowme(img, figsize=None, **kwargs):
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    # fig.gca().imshow(img, origin='lower', **kwargs)
    if img.dtype == np.float16:
        img = img.astype(np.float32)
    fig.gca().imshow(img, **kwargs)
    fig.gca().set_aspect('equal', 'datalim')
    fig.gca().set_position([0, 0, 1, 1])
    return fig

class SelectFromCollection(object):
    """
    Code taken directly from Matplotlib examples site!

    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


## think about how to add this to stackviewer
@DeprecationWarning
class ImEditStack(object):
    def press(self, event):
        sys.stdout.flush()
        if event.key == 'U': # Up
            dimlen = self.stack.shape[self.zdim]
            pt = self.idx[self.zdim]
            if dimlen > 10:
                pt += 1*self.mul
            else:
                pt += 1
            pt %= dimlen
            self.idx[self.zdim] = pt
        elif event.key == 'D': # Down
            dimlen = self.stack.shape[self.zdim]
            pt = self.idx[self.zdim]
            if dimlen > 10:
                pt -= 1*self.mul
            else:
                pt -= 1
            pt %= dimlen
            self.idx[self.zdim] = pt
        elif event.key == 'Z': # next active dimension
            self.zdim += 1
            self.zdim %= self.ndim
        elif event.key == 'F': # Faster
            self.mul += 1
        elif event.key == 'S': # Slower
            self.mul -= 1
        elif event.key == 'C':
            self.autocolor = not self.autocolor
        elif event.key == 'P':
            visible = self.fig.gca().images[1].get_visible()
            self.fig.gca().images[1].set_visible(not visible)
        if self.autocolor:
            img = self.stack[tuple(self.idx)]
            mn, mx = img.min(), img.max()
            self.fig.gca().images[0].set_clim(mn, mx)
            print(mn, mx)

        print('idx:', self.idx, 'zdim:', self.zdim, 'mul:', self.mul, 'autocolor:', self.autocolor)
        self.fig.gca().images[0].set_data(self.stack[tuple(self.idx)])
        self.fig.gca().images[1].set_data(self.overlay[self.idx[-1]])
        self.fig.canvas.draw()

    def on_key_press(self, event):
       if not self.active_label and event.key in ['1', '2', '3', '4', '5']:
           self.active_label = int(event.key)
           print("HELD")

    def on_key_release(self, event):
       if event.key in ['1', '2', '3', '4', '5']:
           self.active_label = None
           print("RELEASED")

    def draw(self, event):
        if self.active_label: # self.active_label:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            z = self.idx[-1]
            if not np.all(self.overlay[z,y,x] == self.label_cmap[self.active_label-1]):
                self.overlay[z,y,x] = self.label_cmap[self.active_label-1]
                self.fig.gca().images[1].set_data(self.overlay[z])

            # print('doit: ', x,y)
            # self.fig.canvas.draw()

    def __init__(self, stack):
        # fig = plt.figure()
        # fig.gca().imshow(stack[z])

        self.zdim = 0
        self.idx = np.array([0]*(stack.ndim-2)) # TODO: this will give a bug when we have 3channel color imgs.
        self.stack = stack
        self.overlay = np.zeros(stack.shape[-3:] + (4,), dtype='float')
        # self.overlay[:100, :50] = 4
        # self.overlay[100:, :50] = 1
        self.alpha_overlay = 0.75

        fig = imshowme(stack[tuple(self.idx)])
        
        fig.canvas.mpl_connect('key_press_event', self.press)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        fig.canvas.mpl_connect('motion_notify_event', self.draw)
        fig.gca().imshow(self.overlay[self.idx[-1]])
        fig.gca().images[1].set_clim(0,5)

        print(fig, self.idx)
        self.fig = fig
        # self.cid = cid
        self.mul = 4
        self.ndim = stack.ndim-2
        self.autocolor = True
        self.active_label = None
        self.label_cmap = np.array([[1,0,0,1], [0,1,0,1], [0,0,0,0], [0,0,1,1]])

@DeprecationWarning
class ArrowToggler(object):
    def __init__(self, iss, tr, draw_arrows, clear_arrows):
        self.cid = iss.fig.canvas.mpl_connect('key_press_event', self.change_arrows)
        self.arrows_on = False
        self.iss = iss
        self.tr  = tr
        self.draw_arrows = draw_arrows
        self.clear_arrows = clear_arrows

    def change_arrows(self, event):
        self.arrows_on = not self.arrows_on
        zi = self.iss.idx[0]
        if event.key == 'a':
            print('toggle arrows', event.key)
            if self.arrows_on:
                zi = min(zi, self.iss.stack.shape[0]-2)
                print(zi)
                self.draw_arrows(self.iss, self.tr.al[zi])
                print("OK")
            else:
                self.clear_arrows(self.iss)
                print("OFF")
            self.iss.fig.canvas.draw()
