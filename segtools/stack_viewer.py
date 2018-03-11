## simple stack viewer

import sys
import numpy as np
import matplotlib.pyplot as plt

class ImshowStack(object):
    """
    Quick and dirty viewer for ndarrays with ndim>2.
    'U' = Up
    'D' = Down
    'F' = Faster
    'S' = Slower
    'C' = toggle auto contrast
    'Z' = change active dimension
    'p' = toggle pan/zoom (ctrl + drag = zoom)
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
        elif event.key == 'C':
            self.autocolor = not self.autocolor
        if self.autocolor:
            img = self.stack[tuple(self.idx)]
            mn, mx = img.min(), img.max()
            self.fig.gca().images[0].set_clim(mn, mx)
            print(mn, mx)

        print('idx:', self.idx, 'zdim:', self.zdim, 'mul:', self.mul, 'autocolor:', self.autocolor)
        self.fig.gca().images[0].set_data(self.stack[tuple(self.idx)])
        self.fig.canvas.draw()

    def __init__(self, stack, xydims=[-1,-2]):
        self.zdim = 0
        # TODO: this will give a bug when we have 3channel color imgs.
        dims = range(stack.ndim)
        xdim = dims[xydims[0]]
        ydim = dims[xydims[1]]

        self.idx = np.array([0]*(stack.ndim-2))
        self.stack = stack
        # self.overlay[:100, :50] = 4
        # self.overlay[100:, :50] = 1

        fig = imshow_plus(stack[tuple(self.idx)])
        
        fig.canvas.mpl_connect('key_press_event', self.press)
        fig.gca().imshow(self.overlay[self.idx[-1]])

        print(fig, self.idx)
        self.fig = fig
        self.mul = 4
        self.ndim = stack.ndim-2
        self.autocolor = True

def imshow_plus(img, **kwargs):
    fig = plt.figure() #figsize=(8,6))
    fig.gca().imshow(img, origin='lower', **kwargs)
    fig.gca().set_aspect('equal', 'datalim')
    fig.gca().set_position([0, 0, 1, 1])
    return fig
