import numpy as np
import matplotlib.pyplot as plt

import spimagine

from . import patchmaker
from . import nhl_tools


def get_slices_from_transform(img, tcube):
    """
    specify a spimagine.TransformData and full-sized image
    get the image cube inside the bounding box.
    """
    zhw, yhw, xhw = np.array(img.shape)/2
    tcube.bounds
    xmin = int((1 + tcube.bounds[0])*xhw)
    xmax = int((1 + tcube.bounds[1])*xhw)
    ymin = int((1 + tcube.bounds[2])*yhw)
    ymax = int((1 + tcube.bounds[3])*yhw)
    zmin = int((1 + tcube.bounds[4])*zhw)
    zmax = int((1 + tcube.bounds[5])*zhw)
    slt = slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax)
    return slt

def transform2slices(trans, shape):
    """
    specify a spimagine.TransformData and full-sized image
    get the image cube inside the bounding box.
    """
    zhw, yhw, xhw = np.array(shape)/2
    trans.bounds
    xmin = int((1 + trans.bounds[0])*xhw)
    xmax = int((1 + trans.bounds[1])*xhw)
    ymin = int((1 + trans.bounds[2])*yhw)
    ymax = int((1 + trans.bounds[3])*yhw)
    zmin = int((1 + trans.bounds[4])*zhw)
    zmax = int((1 + trans.bounds[5])*zhw)
    slt = slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax)
    return slt

def get_cube_from_transform(img, tcube):
    """
    specify a spimagine.TransformData and full-sized image
    get the image cube inside the bounding box.
    """
    zhw, yhw, xhw = np.array(img.shape)/2
    tcube.bounds
    xmin = int((1 + tcube.bounds[0])*xhw)
    xmax = int((1 + tcube.bounds[1])*xhw)
    ymin = int((1 + tcube.bounds[2])*yhw)
    ymax = int((1 + tcube.bounds[3])*yhw)
    zmin = int((1 + tcube.bounds[4])*zhw)
    zmax = int((1 + tcube.bounds[5])*zhw)
    cube = img[zmin:zmax, ymin:ymax, xmin:xmax]
    return cube

def curate_nocrop(nhl, img, hyp, w):
    def shownuc(n):
        img2 = img.copy()
        mask = lib.mask_nhl([n], hyp)
        img2[mask] = img2.max() * 1.5
        update_spim(w, 0, img2)
        ans = input("How many nuclei do you see? :: ")
        # ans = (ans, w.transform.maxVal)
        return ans

    biganno = ['no idea' for _ in nhl]

    i=0
    while i < len(nhl):
        ans = shownuc(nhl[i])
        if ans == 'q':
            break
        elif ans == 'k':
            print("Undo...")
            i -= 1
        else:
            biganno[i] = ans
            i += 1
    return biganno

def curate_nhl(nhl, img, hyp, pp, w=None):
    img = np.pad(img, pp, mode='constant')
    hyp = np.pad(hyp, pp, mode='constant')
    
    def crops(i):
        nuc = nhl[i]
        ss = lib.nuc2slices_centroid(nuc, pp, shift=pp)
        img_crop = img[ss].copy()
        hyp_crop = hyp[ss].copy()
        mask = hyp_crop == nuc['label']
        img_crop[mask] *= 2.0
        return img_crop, hyp_crop

    def nextnuc(i):
        imgc, hypc = crops(i)
        update_spim(w, 0, imgc)
        ans = input("How many nuclei do you see? :: ")
        ans = (ans, w.transform.maxVal)
        return ans

    biganno = ['no idea' for _ in nhl]

    imgc, hypc = crops(0)
    if w is None:
        w = spimagine.volshow([imgc, hypc])
    w.transform.setMax(imgc.max())
    ans = nextnuc(0)

    if ans == 'q':
        return None
    else:
        biganno[0] = ans
    i = 1
    while i < len(nhl):
        ans = nextnuc(i)
        if ans == 'q':
            break
        elif ans == 'k':
            print("Undo...")
            i -= 1
        else:
            biganno[i] = ans
            i += 1
    return w, biganno

def update_spim(w, i, cube):
    w.glWidget.dataModel[i][...] = cube
    w.glWidget.dataPosChanged(i)

def mk_quat(m):
    "this is the boundary from our data (z,y,x) indicies. To spimagine data: (x,y,z) inds."
    m = m[:, [2, 1, 0]]
    w = np.sqrt(1.0 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    w4 = (4.0 * w)
    x = (m[2, 1] - m[1, 2]) / w4
    y = (m[0, 2] - m[2, 0]) / w4
    z = (m[1, 0] - m[0, 1]) / w4
    # TODO: Turn this into a quaternion OUTSIDE the drawing namespace!
    return (w, x, y, z)

def moveit(w):
    dm = w.glWidget.dataModel
    auto = True
    auto_percentile = 100.0
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    alpha_i = 0
    while True:
        cmd = input("Command: ")
        if cmd == 'q':
            break
        elif cmd == 'x':
            # quatRot = spimagine.Quaternion(0.5,-0.5,0.5,0.5)
            w.transform.addRotation(np.pi*0.25, 1., 0., 0.)
        elif cmd == 'y':
            w.transform.addRotation(np.pi*0.25, 0., 1., 0.)
        elif cmd == 'z':
            w.transform.addRotation(np.pi*0.25, 0., 0., 1.)
        elif cmd == 'reset':
            # quatRot = spimagine.Quaternion(0.5,-0.5,0.5,0.5)
            w.transform.setRotation(0, 1., 0., 0.)
        elif cmd == 'l':
            p = w.transform.toTransformData().dataPos
            if p < dm.size()[0]-1:
                w.transform.setPos(p+1)
                if auto:
                    val = np.percentile(dm[p+1], auto_percentile)
                    w.transform.setMax(val)
        elif cmd == 'h':
            p = w.transform.toTransformData().dataPos
            if p >= 1:
                w.transform.setPos(p-1)
                if auto:
                    val = np.percentile(dm[p-1], auto_percentile)
                    w.transform.setMax(val)
        elif cmd == 'a':
            tr = w.transform.toTransformData()
            alpha_i += 1
            alpha_i %= len(alphas)
            tr.alphaPow = alphas[alpha_i]
            # tr.alphaPow %= 1.0
            w.transform.fromTransformData(tr)
        elif cmd == 't':
            tr = w.transform.toTransformData()
            print(tr)
        elif cmd == 'p':
            p = w.transform.isPerspective
            w.transform.setPerspective(not p)
        elif cmd == 'c':
            print(w.transform.maxVal)
        elif cmd[0] == 'c':
            tr = w.transform.toTransformData()
            tr.maxVal = float(cmd[2:])
            w.transform.fromTransformData(tr)
        elif cmd == 'auto off':
            auto = False
        elif cmd[:4] == 'auto':
            auto = True
            auto_percentile = float(cmd[6:])


def render_rgb_still(hypRGB, w=None, transform=None, fname="sceneRGB.png"):
    if w is None:
        w = spimagine.volshow(hypRGB[...,0], interpolation='nearest', cmap='grays', raise_window=False)
    
    if transform:
        w.transform.fromTransformData(transform)
    
    update_spim(w,0,hypRGB[...,0])
    w.saveFrame('img0.png')
    update_spim(w,0,hypRGB[...,1])
    w.saveFrame('img1.png')
    update_spim(w,0,hypRGB[...,2])
    w.saveFrame('img2.png')
    
    chan1 = io.imread('img0.png')[...,0]
    chan2 = io.imread('img1.png')[...,0]
    chan3 = io.imread('img2.png')[...,0]
    flat  = np.stack((chan1, chan2, chan3), axis=-1)
    io.imsave(fname, flat)
    return 


def comboview(img3d, axis=0, hyp=None, tform=None):
    # axis x,y,z = 2,1,0
    # show 2d and 3d views
    # update those views as we move

    # zcur = 50

    # def press(event):
    #     print('press', event.key)
    #     sys.stdout.flush()
    #     if event.key == '1':
    #         # visible = xl.get_visible()
    #         # xl.set_visible(not visible)
    #         fig.gca().images[0].set_visible()
    #         # fig.canvas.draw()
    #     if event.key == 'v':
    #         xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
    #         print(img[yi, xi])
    #     if event.key == 'up':
    #         zcur += 1
    #         fig.gca().image[0].set_data(img3d[zcur])
    #     if event.key == 'down':
    #         zcur -= 1
    #         fig.gca().image[0].set_data(img3d[zcur])


    # def onclick(event):
    #     print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #           (event.button, event.x, event.y, event.xdata, event.ydata))
    #     xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
    #     # define slice of img3d with min that has a max size of 200x200 on it's non-z axes
    #     xmn,xmx = max(0,xi-r), min(imgproj.shape[1], xi+r)
    #     ymn,ymx = max(0,yi-r), min(imgproj.shape[0], yi+r)
    #     if xmx - xmn < 50 or ymx-ymn < 50:
    #         return
    #     slc = [slice(ymn,ymx), slice(xmn,xmx)]
    #     slc.insert(axis, slice(None))
    #     cube = img3d[slc].copy()
    #     print(cube.shape)

    #     # define container for data that has exactly 200x200 on it's non-z axes
    #     shp = [200]*3
    #     shp[axis] = img3d.shape[axis]
    #     fullcube = np.zeros(shp, np.float32)
    #     print(fullcube.shape)
    #     a,b,c = cube.shape
        
    #     # put cube inside of container and update 3d view
    #     fullcube[:a, :b, :c] = cube
    #     w.glWidget.renderer.update_data(fullcube)
    #     w.glWidget.refresh()


    # setup 2d figure
    midspot = img3d.shape[axis]//2
    slc = [slice(None)]*3
    slc[axis] = slice(midspot, midspot + 30)
    # slc[axis] = slice(zcur, zcur+1)
    imgproj = img3d[slc].max(axis)
    fig = imshowme(imgproj)

    # spimagine.config.__DEFAULT_SPIN_AXIS__ = 1 #2-axis ## because of opencl axis inversion
    
    # setup 3d view
    r = 100
    slc = [slice(0,2*r)]*3
    slc[axis] = slice(None)
    w = spimagine.volshow(img3d[slc], raise_window=False, interpolation="nearest")
    if tform:
        w.transform.fromTransformData(tform)
    w.glWidget.refresh()

    # define click|press events
    click_genf = onclick_gen(img3d, w, axis, imgproj.shape, r)
    # press_genf = press_gen(fig, imgproj)
    cid = fig.canvas.mpl_connect('button_press_event', click_genf)
    # cid = fig.canvas.mpl_connect('key_press_event', press)
    return fig, w