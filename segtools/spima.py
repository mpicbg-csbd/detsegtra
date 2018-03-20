import numpy as np
import matplotlib.pyplot as plt

from . import patchmaker
from . import lib


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

def curate_nhl(w, nhl, img, hyp, pp):
    def crops(i):
        nuc = nhl[i]
        ss = lib.nuc2slices_centroid(nuc, pp//2)
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
    return biganno


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
