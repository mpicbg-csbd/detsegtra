## Additional Matplotlib and Spimagine functionality
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
plt.ion()

import spimagine
import sys

# ---- interactive 2d + 3d viewing ----

def onclick_gen(img3d, w, axis, img2dshape, r = 100):
    d1len,d2len = img2dshape
    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        # define slice of img3d with min that has a max size of 200x200 on it's non-z axes
        xmn,xmx = max(0,xi-r), min(d2len, xi+r)
        ymn,ymx = max(0,yi-r), min(d1len, yi+r)
        if xmx - xmn < 50 or ymx-ymn < 50:
            return
        slc = [slice(ymn,ymx), slice(xmn,xmx)]
        slc.insert(axis, slice(None))
        cube = img3d[slc].copy()
        print(cube.shape)

        # define container for data that has exactly 200x200 on it's non-z axes
        shp = [200]*3
        shp[axis] = img3d.shape[axis]
        fullcube = np.zeros(shp, np.float32)
        print(fullcube.shape)
        a,b,c = cube.shape
        
        # put cube inside of container and update 3d view
        fullcube[:a, :b, :c] = cube
        # w.glWidget.renderer.update_data(fullcube)
        # w.glWidget.refresh()
        w.glWidget.dataModel[0][...] = cube
        w.glWidget.dataPosChanged(0)

    return onclick

def onclick_gen4d(img4d, w, axis, img2dshape, r = 100):
    d1len,d2len = img2dshape
    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        # define slice of img3d with min that has a max size of 200x200 on it's non-z axes
        xmn,xmx = max(0,xi-r), min(d2len, xi+r)
        ymn,ymx = max(0,yi-r), min(d1len, yi+r)
        if xmx - xmn < 50 or ymx-ymn < 50:
            return
        slc = [slice(ymn,ymx), slice(xmn,xmx)]
        slc.insert(axis, slice(None))
        cube = img3d[:, slc].copy()
        print(cube.shape)

        # define container for data that has exactly 200x200 on it's non-z axes
        shp = [200]*3
        shp[axis] = img3d.shape[axis]
        fullcube = np.zeros(shp, np.float32)
        print(fullcube.shape)
        a,b,c = cube.shape
        
        # put cube inside of container and update 3d view
        fullcube[:a, :b, :c] = cube
        # w.glWidget.renderer.update_data(fullcube)
        # w.glWidget.refresh()
        w.glWidget.dataModel[0][...] = cube
        w.glWidget.dataPosChanged(0)

    return onclick

def press_gen(fig, img):
    def press(event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == '1':
            # visible = xl.get_visible()
            # xl.set_visible(not visible)
            fig.gca().images[0].set_visible()
            # fig.canvas.draw()
        if event.key == 'v':
            xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
            print(img[yi, xi])
    return press

def curate_2imgs(img3d, pimg3d):
    # show 2d and 3d views
    # update those views as we move
    imgproj1 = img3d[300:310].max(0)
    imgproj2 = pimg3d[300:310].max(0)

    fig = imshowme(imgproj2, alpha=0.5)
    fig.gca().imshow(imgproj1, alpha=0.5)
    
    # define keys
    # click_genf = onclick_gen(img3d)
    # press_genf = press_gen(fig, imgproj)
    # cid = fig.canvas.mpl_connect('button_press_event', click_genf)
    # cid = fig.canvas.mpl_connect('key_press_event', press_genf)
    return fig

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

class TestReader(object):
    def __init__(self):
        inp = input('reading your shit')
        print(inp)

class ImshowStack(object):
    """
    Quick and dirty viewer for ndarrays with ndim>2.
    'U' = Up
    'D' = Down
    'F' = Faster
    'S' = Slower
    'A' = toggle auto contrast
    'Z' = change active dimension
    """
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
        elif event.key == 'A':
            self.autocolor = not self.autocolor
        elif event.key == 'V':
            assert False
        # elif event.key == 'W':
        #     self.w = int(input('gimme a w: '))
        
        if self.autocolor:
            img = self.stack[tuple(self.idx)]
            mn, mx = img.min(), img.max()
            self.fig.gca().images[0].set_clim(mn, mx)
            print(mn, mx)

        print('idx:', self.idx, 'zdim:', self.zdim, 'mul:', self.mul, 'autocolor:', self.autocolor)
        
        if self.w > 0:
            zpos = self.idx[1]
            # a = max(zpos-self.w, 0)
            b = min(zpos+self.w, self.stack.shape[1])
            ss = list(self.idx)
            ss[1] = slice(zpos, b)
            data = self.stack[tuple(ss)]
            if decay:
                decay = np.exp(-np.arange(data.shape[0])/2).reshape((-1,1,1))
                data = (data*decay).mean(0)
        else:
            ss = tuple(self.idx)
            data = self.stack[ss]
        
        self.fig.gca().images[0].set_data(data)
        self.fig.canvas.draw()

    def __init__(self, stack, customclick=None, colorchan=False, w=0, decay=False, norm=True):
        if type(stack)==list:
            stack = np.stack(stack).astype(np.float)
            print(stack.dtype)
        self.zdim = 0
        # TODO: this will give a bug when we have 3channel color imgs.
            
        if colorchan:
            if stack.shape[-1]==2:
                stack = np.stack([stack[...,0], 0.5*stack[...,1], 0.5*stack[...,1]], axis=-1)
            self.idx = np.array([0]*(stack.ndim-3))
            self.ndim = stack.ndim-3
        else:
            self.idx = np.array([0]*(stack.ndim-2))
            self.ndim = stack.ndim-2
        self.stack = stack
        # self.overlay = np.zeros(stack.shape[-3:] + (4,), dtype='float')
        # self.overlay[:100, :50] = 4
        # self.overlay[100:, :50] = 1

        fig = imshowme(stack[tuple(self.idx)])
        
        fig.canvas.mpl_connect('key_press_event', self.press)
        # fig.gca().imshow(self.overlay[self.idx[-1]])

        print(fig, self.idx)
        self.fig = fig
        self.mul = 1
        self.autocolor = True
        self.w = w

class Stack(object):
    """
    Quick and dirty viewer for ndarrays with ndim>2.
    Hold down the number of the dimension you want to cycle, eg '1', '2', ...
    Move forwards/backwards with k/j
    'F' = Faster
    'S' = Slower
    'A' = toggle auto contrast
    """
    def press(self, event):
        sys.stdout.flush()
        if event.key == 'i': # Up
            dimlen = self.stack.shape[self.zdim]
            pt = self.idx[self.zdim]
            if dimlen > 10:
                pt += 1*self.mul
            else:
                pt += 1
            pt %= dimlen
            self.idx[self.zdim] = pt
        elif event.key == 'j': # Down
            dimlen = self.stack.shape[self.zdim]
            pt = self.idx[self.zdim]
            if dimlen > 10:
                pt -= 1*self.mul
            else:
                pt -= 1
            pt %= dimlen
            self.idx[self.zdim] = pt
        elif event.key in {'1', '2', '3'}:
            self.zdim = int(event.key)-1
            print('idx:', self.idx, 'zdim:', self.zdim, 'mul:', self.mul, 'autocolor:', self.autocolor)
        elif event.key == 'F': # Faster
            self.mul += 1
        elif event.key == 'S': # Slower
            self.mul -= 1
        elif event.key == 'A':
            self.autocolor = not self.autocolor
        elif event.key == 'V':
            assert False
        # elif event.key == 'W':
        #     self.w = int(input('gimme a w: '))
        
        if self.autocolor:
            img = self.stack[tuple(self.idx)]
            mn, mx = img.min(), img.max()
            self.fig.gca().images[0].set_clim(mn, mx)
            # print(mn, mx)

        # print('idx:', self.idx, 'zdim:', self.zdim, 'mul:', self.mul, 'autocolor:', self.autocolor)
        
        if self.w > 0:
            zpos = self.idx[1]
            # a = max(zpos-self.w, 0)
            b = min(zpos+self.w, self.stack.shape[1])
            ss = list(self.idx)
            ss[1] = slice(zpos, b)
            data = self.stack[tuple(ss)]
            if decay:
                decay = np.exp(-np.arange(data.shape[0])/2).reshape((-1,1,1))
                data = (data*decay).mean(0)
        else:
            ss = tuple(self.idx)
            data = self.stack[ss]
        
        self.fig.gca().images[0].set_data(data)
        self.fig.canvas.draw()

    def __init__(self, stack, customclick=None, colorchan=False, w=0, decay=False, norm=True):
        if type(stack)==list:
            stack = np.stack(stack).astype(np.float)
            print(stack.dtype)
        self.zdim = 0
        # TODO: this will give a bug when we have 3channel color imgs.
            
        if colorchan:
            if stack.shape[-1]==2:
                stack = np.stack([stack[...,0], 0.5*stack[...,1], 0.5*stack[...,1]], axis=-1)
            self.idx = np.array([0]*(stack.ndim-3))
            self.ndim = stack.ndim-3
        else:
            self.idx = np.array([0]*(stack.ndim-2))
            self.ndim = stack.ndim-2
        self.stack = stack
        # self.overlay = np.zeros(stack.shape[-3:] + (4,), dtype='float')
        # self.overlay[:100, :50] = 4
        # self.overlay[100:, :50] = 1

        fig = imshowme(stack[tuple(self.idx)])
        
        fig.canvas.mpl_connect('key_press_event', self.press)
        # fig.gca().imshow(self.overlay[self.idx[-1]])

        print(fig, self.idx)
        self.fig = fig
        self.mul = 1
        self.autocolor = True
        self.w = w

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
