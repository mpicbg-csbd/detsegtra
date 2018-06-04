import numpy as np
from scipy.misc import imresize
import skimage.transform as tform
from scipy.ndimage import label, zoom, rotate, distance_transform_edt
import random


def warp_onechan(img, delta):
    """
    warp img according to random gaussian vector field.
    img may have arbitrary number of channels, or be multiple images stacked together.
    Interpolate between a square grid of w**2 random gaussian vectors with standard deviation = stddev
    """
    deltax, deltay = delta[0], delta[1]
    a,b = img.shape
    deltax = imresize(deltax, size=(a,b), mode='F')
    deltay = imresize(deltay, size=(a,b), mode='F')
    a1,b1 = deltax.shape
    # deltax = zoom(deltax, (a/a1,b/b1), order=3)
    # deltay = zoom(deltay, (a/a1,b/b1), order=3)
    # MAX GRADIENTS should be less than 1 to avoid folds
    # dxdx = np.max(np.diff(deltax, axis=0))
    # dydx = np.max(np.diff(deltax, axis=1))
    # dxdy = np.max(np.diff(deltay, axis=0))
    # dydy = np.max(np.diff(deltay, axis=1))
    # print("MAX GRADS", dxdx, dydx, dxdy, dydy)
    delta_big = np.stack((deltax, deltay), axis=0)
    coords = np.indices(img.shape)
    newcoords = delta_big + coords
    res = tform.warp(img, newcoords, order=1, mode='reflect')
    return res, delta_big, coords

def warp_multichan(img, delta):
    a,b,c = img.shape
    stack = img.copy()
    for i in range(c):
        stack[:,:,i] = warp_onechan(img[:,:,i], delta)[0]
    return stack

def plot_vector_field(img):
    """
    only designed to be the right scale for smooth warps of roughly 500^2 image patches.
    VERY FRUSTRATING THAT... images have dimensions [y,x] = [vertical,horizontal]
    and the vertical axis is always plotted DOWNWARDS!!! y goes from 0 (top) to
    y_max at the bottom! This is the opposite of all other plots...
    """
    n = 10
    delta = np.random.normal(loc=0, scale=5, size=(2,3,3))
    res, delta, coords = warp_onechan(img, delta=delta, twolabel=True)
    plt.figure()
    plt.imshow(img[::-1])
    plt.figure()
    plt.imshow(res[::-1])
    plt.figure()
    plt.quiver(delta[1,::n,::n],
                delta[0,::n,::n],
                headlength=2,
                headwidth=2,
                headaxislength=3)
    plt.streamplot(coords[1,0,::n]/n,
                   coords[0,::n,0]/n,
                   delta[1,::n,::n],
                   delta[0,::n,::n])
    plt.savefig('warp_plot.png')

def explore_warps(img):
    ss = [2]*30
    ws = [10]
    big = np.zeros((len(ss), len(ws)) + img.shape)
    x,y = 0,0
    for s in ss:
        y=0
        for w in ws:
            # res = warp_gaussian(img, stdev=s, w=w)
            res = warp_onechan(img)
            big[x, y] = res
            y+=1
        x+=1
    return big

def explore_warps_multisize(img):
    """
    when you want to make a 2d tiling of images with different (but similar) sizes.
    """
    x,y = 0,0
    a,b = img.shape
    big = np.zeros(shape=(a*7, b*11))
    print(big.shape)
    # assert 0
    for s in np.linspace(0, 3, 4):
        y=0
        for w in range(5,15,3):    
            res, sumy, sumx = warp(img, mean=0, stdev=s, w=w)
            a2,b2 = res.shape
            print(a2,b2)
            print()
            big[x:x+a2, y:y+b2] = res
            y+=b
        x+=a
    return big

