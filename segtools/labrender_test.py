## all must do:
%gui qt5

import numpy as np
import vispy
# vispy.use('pyqt5')
import tifffile
import napari
from scipy.ndimage import label
from matplotlib import colors
import matplotlib.pyplot as plt

import spimagine
from spimagine.volumerender.volumerender import VolumeRenderer

from segtools.StackVis import StackVis
from segtools.numpy_utils import normalize3, perm2, stak
from segtools import render
from segtools import spima
from segtools import color


def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)


## section 1: render the flat embryo with nice colormap at various threshold levels

embryo = imread('/Users/broaddus/Dropbox/SPC0_TM0037_CM0_CM1_CHN00_CHN01.fusedStack.tif')
embryo = normalize3(embryo,0,100)

def lab2rgb(lab):
  # cmap_rand = color.rand_cmap_uwe()
  # np.save('/Users/broaddus/Desktop/Projects/friday_seminar/cmap_rand.npy',cmap_rand)
  # lab = label(lab)[0]; m=lab==0; lab%=256; lab[m]=0
  cmap_rand = np.load('/Users/broaddus/Desktop/Projects/friday_seminar/cmap_rand.npy')
  return cmap_rand[lab.flat].reshape(lab.shape + (3,))

def f(x=60):
  eseg = embryo[:60] > np.percentile(embryo[:60],x)
  elab = label(eseg)[0]
  # elab = render.first_nonzero(elab,0).astype(np.uint64)
  m=elab==0;elab%=255;elab+=1;elab[m]=0
  return elab #lab2rgb(elab)

stack = stak(f(50+29),f(55+29),f(60+29),f(65+29),f(70+29))

iss = StackVis(stack.astype(np.uint8))
cmap_rand = np.load('/Users/broaddus/Dropbox/phd/finished/2019_02_22 -- Friday Seminar Long/friday_seminar_figures/cmap_rand.npy')
iss.image.cmap = vispy.color.Colormap(cmap_rand,interpolation='linear') ## works
# iss.image.cmap = vispy.color.Colormap(cmap_rand,interpolation='zero') ## fails at zero, bug?

results = """
The high threshold looks best. stack[4].
"""

## section 2: explore spimagine rendering fly embryo and segmentation

embryo = imread('/Users/broaddus/Dropbox/SPC0_TM0037_CM0_CM1_CHN00_CHN01.fusedStack.tif')
embryo = normalize3(embryo,0,100)
seg = label(embryo > np.percentile(embryo,98))[0]
def modlab(lab,max=255):
  lab = lab.copy()
  m=lab==0;lab%=max;lab+=1;lab[m]=0
  return lab
lab = modlab(seg)

# new_cmap = color.rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=True)
# cmap_pastel = color.pastel_colors_RGB(brightness=.8)
# cmap_pastel = colors.ListedColormap(cmap_pastel)

w = spimagine.volshow(embryo,stackUnits=[1,1,5],interpolation='nearest',autoscale=False)

tf = spimagine.TransformData(quatRot = spimagine.Quaternion(-0.006744930241088288,-0.5834835983350474,-0.810120559049137,0.05662222129352936), zoom = 1.569981022187107,
                             dataPos = 0,
                             minVal = 1e-06,
                             maxVal = 0.5373745864277504,
                             gamma= 1.0,
                             translate = np.array([0, 0, 0]),
                             bounds = np.array([-1.  ,  1.  , -1.  ,  1.  , -0.03,  1.  ]),
                             isBox = True,
                             renderMode= 0,
                             alphaPow = 0.1,
                             isSlice = False,
                             slicePos = 0,
                             sliceDim = 0
                             )

w.transform.fromTransformData(tf)
# w.transform.setMax(.5373)
# w.transform.setAlphaPow(0.1)
raw_rendered = w.glWidget.renderer.output.copy()

spima.update_spim(w,0,lab)
w.transform.setMax(lab.max())
# w.glWidget._set_colormap_array(cmap_rand)
w.transform.setRenderMode(2) ## first-nonzero-hit rendering exists!
lab_rendered = w.glWidget.renderer.output.copy()
lab_rendered = np.round(lab_rendered*255).astype(np.int)
# lab_rendered = modlab(lab_rendered)

viewer = napari.view_image(raw_rendered)
viewer.add_labels(lab_rendered)


## section: now compare against render module orthogonal proj

import render as ren

embryo = imread('/Users/broaddus/Dropbox/SPC0_TM0037_CM0_CM1_CHN00_CHN01.fusedStack.tif')
embryo = normalize3(embryo,0,100)
seg = label(embryo > np.percentile(embryo,98))[0]
def modlab(lab,max=255):
  lab = lab.copy()
  m=lab==0;lab%=max;lab+=1;lab[m]=0
  return lab
lab = modlab(seg)


viewer = napari.view_image(embryo)
rawr,labr = ren.joint_raw_lab_fnzh(embryo[:30],lab[:30])
viewer.add_image(rawr)
viewer.add_labels(labr)

viewer = napari.view_image(embryo)
rawr,labr = ren.joint_raw_lab_fnzh(embryo[-30:],lab[-30:])
viewer.add_image(rawr)
viewer.add_labels(labr)

rawr,labr = ren.joint_raw_lab_fnzh(embryo[:,:,:250],lab[:,:,:250],ax=2)
from scipy.ndimage import zoom
rawr = zoom(rawr,(1,5),order=3)
viewer.add_image(rawr)
labr = zoom(labr,(1,5),order=0)
viewer.add_labels(labr)

rawr,labr = ren.joint_raw_lab_fnzh(embryo[:,:,250:],lab[:,:,250:],ax=2)
from scipy.ndimage import zoom
rawr = zoom(rawr,(1,5),order=3)
viewer.add_image(rawr)
labr = zoom(labr,(1,5),order=0)
viewer.add_labels(labr)

result = """
This looks great! What a nice rendering!
Except for the YZ projections. They look so low res along z that it's hard to tell if the seg is good.
NOTE: lots of little one-pixel segments in the XY view. Also note the large pure-noise segments in the YZ view.
Both the segmentation _and_ the rendering should look much better if we use a pixelwise classifier instead of the raw.
"""

## section: explore render module tribolium

trib = imread('/Users/broaddus/Desktop/Projects/nucleipix_CARE_evaluations/trib/prediction_intensity/nGFP_0.1_0.2_0.5_20_07___1.tif')
lab_trib = label(trib>0.5)[0]
import render as ren

rawr,labr = ren.joint_raw_lab_fnzh(trib,lab_trib)
viewer = napari.view_image(rawr)
viewer.add_labels(labr)

notes = """
This orthogonal rendering works, but is not that impressive.
Trib is just too flat.
Also it suffers from missing raw data of False Negatives (missing nuclei).

"""

## section: perspective render tribolium

trib = imread('/Users/broaddus/Desktop/Projects/nucleipix_CARE_evaluations/trib/prediction_intensity/nGFP_0.1_0.2_0.5_20_07___1.tif')
lab_trib = label(trib>0.5)[0]

lab_trib = lab_trib/lab_trib.max()
w = spimagine.volshow([trib,lab_trib],stackUnits=[1,1,2.5],interpolation='nearest',autoscale=False)

w.transform.setMax(1)
w.transform.setAlphaPow(0.5)

viewer = napari.view_image(trib)

def render_both(w,raw,lab,viewer):
  spima.update_spim(w,0,lab)
  w.
render_both(w,trib,lab_trib,viewer)

results = """
The perspective projection makes tribolium look flat.
NOTE: you have to render with spimagine window at full width! It affects the rendering size!
It's nice that we see all the nuclei at once, but you lose all sense of depth.
TO TRY:
  - I think this rendering needs reflections/shadows/etc to really make the 3D shapes and locations apparent. Maybe try paintera.

"""


## Summary

notes = """
To make the rendering work for lab images we had to change the sampling used in open_cl from linear to const
We also have to use interpolation='nearest' in volshow
we also had to change the kernel for alpha>0 s.t. the value returned is NOT weighted by alphapow.
And we have to turn off the interpolation in open_gl in spimagine.gui_utils.fillTexture2d (interp=False)
NOTE: every time we adjust a slider the volume is re-rendered. even just for adjusting the min/max brightness.


remaining problem: 
- the lower-valued labels appear to be darker. why?
- the thin borders of some shapes have artifacts
  + maybe because delta_pos is too large? and we just miss them?
  + maybe because our Nx,Ny grid is too coarse? does it auto-update with the window size?
- i don't know what w.glWidget.renderer.output_alpha does, and I can't seem to control it from the kernel.
- i can't get the alpha<0 branch of the kernel to work. the z-location of objects changes as you rotate the view.

Tue Oct 29 12:42:12 2019

The rendering in `render_both` seems to be working at the moment!
No forced sleep necessary.
Setting the alphaPow only works via w.glWidget, not in renderer directly.

"""


