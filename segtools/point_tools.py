import numpy as np

## cropping and using points

# def patches_from_img_and_pts(img, pts, xyz_halfwidth=(5,40,40)):
#   img = np.pad(img,[(i,i) for i in xyz_halfwidth],mode='constant')
#   pts = pts + np.array(xyz_halfwidth)
#   def f(img,p):
#     a,b,c = p
#     m,n,o = xyz_halfwidth
#     ss = slice(a-m,a+m), slice(b-n,b+n),slice(c-o,c+o)
#     return img[ss]
#   patches = np.array([f(img,p) for p in pts])
#   return patches

"""
arbitrary dimensionality
given heterogeneous points from various times, sample heterogeneous crops from heterogeneous images.
different ways of doing boundary conditions:
1. pad with value / reflect / etc
2. shift patch s.t. it only takes valid data and doesn't change size.
"""


def patches_from_centerpoints_multi(imglist,centerpoints_list,patchsize,bounds='constant'):
  patches = np.array([patches_from_centerpoints(imglist[i],centerpoints_list[i],patchsize,bounds=bounds) for i in range(len(imglist))])
  return patches

def patches_from_centerpoints(img,centerpoints,patchsize=(32,32),bounds='constant'):
  """
  bounds in 'constant', 'shift'
  """
  patchsize = np.array(patchsize)
  imshape   = np.array(img.shape)
  centerpoints = np.array(centerpoints)
  if centerpoints.ndim==1: centerpoints = centerpoints[None]
  assert patchsize.shape[0]==img.ndim==centerpoints.shape[1]

  ## constant = patches have black borders
  if bounds=='constant':
    img = np.pad(img,[(i,i) for i in patchsize], mode='constant')
    centerpoints = centerpoints + patchsize
  ## shift = patches are recenterd to stay within image volume
  elif bounds=='shift':
    centerpoints = centerpoints.clip(min=patchsize//2, max=np.array(img.shape)-patchsize//2)

  ## patchpoint is top left corner of patch
  patchpoint = centerpoints - patchsize//2
  
  def f(p):
    ss = [slice(p[i],p[i]+patchsize[i]) for i in range(img.ndim)]
    return img[ss]
  patches = np.array([f(p) for p in patchpoint])
  return patches


@DeprecationWarning
def patches_from_points(img_tzyx,pt_tyx):
  t  = pt_tyx[0]
  yx = pt_tyx[1:]
  bounds = np.array(img_tzyx.shape[-2:])
  yx = yx.clip(min=(300,300),max=bounds-300)
  start,stop = yx-300,yx+300
  print(start,stop)
  return img_tzyx[t,:,start[0]:stop[0],start[1]:stop[1]]

def trim_images_from_pts(pts,*imgs):
  mn = pts.min(0)
  mx = pts.max(0)
  mn2 = np.maximum(mn-(5,15,15),0)
  mx2 = mx+(5,15,15)
  ss = slice(mn2[0],mx2[0]), slice(mn2[1],mx2[1]), slice(mn2[2],mx2[2])
  pts2 = pts-mn2
  imgs2 = [x[ss] for x in imgs]
  return pts2,imgs2

def pts2ss(start,stop): return tuple(slice(a,b) for a,b in zip(start,stop))

def ptstak(pts):
  def f(i,x):
    n,d = x.shape
    ts  = np.zeros((n,1))+i
    return np.concatenate([ts,x],axis=1)
    # x = x[:,[0,0,1]]
    # x[:,0] = i
    # return x
  return np.concatenate([f(i,x) for i,x in enumerate(pts)])


def trim_images_from_pts2(pts):
  "to transform pts, just subtract mn2. to crop image, apply ss."
  mn = pts.min(0)
  mx = pts.max(0)
  mn2 = np.maximum(mn-(5,15,15),0)
  mx2 = mx+(5,15,15)
  ss = slice(mn2[0],mx2[0]), slice(mn2[1],mx2[1]), slice(mn2[2],mx2[2])
  return mn2,ss