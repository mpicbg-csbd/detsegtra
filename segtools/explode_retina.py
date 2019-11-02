from spimagine.volumerender.volumerender import VolumeRenderer
import tifffile
from numpy_utils import normalize2, perm2
from pathlib import Path
import numpy as np
from skimage.measure import regionprops
import spimagine
import skimage.io as io
from math import floor
from path_utils import mkdir

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)

savedir = Path('../../devseg_data/fish_train/d021/')

def load():
  global img, lab, rend
  img = imread(savedir / 'img.tif')
  img = img[0].transpose((0,2,3,1))
  img = normalize2(img,2,99.8,(0,1,2))
  img = img.astype(np.float32)
  img[...,2] = 0

  lab = imread(savedir/'lab.tif')

  rend = VolumeRenderer((500,500))
  rend.set_units([1.,1.,2])
  rend.set_projection(spimagine.mat4_perspective(70,1.,0.1,10))
  T = spimagine.mat4_translate(0,0,-2)
  S = spimagine.mat4_scale(1,1,1)
  # R = 
  rend.set_modelView(T @ S)
  rend.alphaPow = 0.3

def explode_lab():
  nhl = regionprops(lab)
  # nhl = sorted(nhl,key=lambda n: n.area)
  a,b,c,d = img.shape
  newimg = np.zeros((2*a,2*b,2*c,d))

  for k in range(200):
    # img2 = img.copy()
    mx,nx,my,ny,mz,nz = 0,0,0,0,0,0
    ss1s, ss2s = [], []
    for i,n in enumerate(nhl):
      # if i%10==0: continue
      z1,y1,x1,z2,y2,x2 = n.bbox
      ss1 = slice(z1,z2), slice(y1,y2), slice(x1,x2)
      r = 1 + k/200
      z = int(r*z1) # 100
      y = int(r*y1) # (i // int(np.sqrt(len(nhl))))*20 + 20
      x = int(r*x1) # i % int(np.sqrt(len(nhl)))*20 + 20
      ss2 = slice(z,z+z2-z1), slice(y,y+y2-y1), slice(x,x+x2-x1)
      # ss1s.append(ss1)
      # ss2s.append(ss2)
      mx,my,mz = max(mx,x+x2-x1),max(my,y+y2-y1),max(mz,z+z2-z1)
      nx,ny,nz = min(nx,x),min(ny,y),min(nz,z)
      newimg[ss2][n.filled_image] = img[ss1][n.filled_image] / np.percentile(img[ss1][n.filled_image],99.8,axis=0,keepdims=True)
    res = render_3chan(newimg)
    io.imsave('movie/test{:03d}.png'.format(k), res)
    newimg[...] = 0

def explode_lab2():
  """
  explode retina cells away from each other, then rotate in various ways.
  """
  load()
  nhl = regionprops(lab)
  allimgs = []
  # nhl = sorted(nhl,key=lambda n: n.area)

  def f(n):
    z1,y1,x1,z2,y2,x2 = n.bbox
    ss1 = slice(z1,z2), slice(y1,y2), slice(x1,x2)
    crop = img[ss1].copy()
    crop[~n.filled_image] = 0
    crop = crop / np.percentile(crop,99.9,axis=(0,1,2),keepdims=True)
    return locals()

  crops = [f(n) for n in nhl]

  def g(n):
    z1,y1,x1,z2,y2,x2 = n.bbox
    r = 1 + k/200
    z_1 = int(r*z1) + int(90*(1-k/200)) # 100
    y_1 = int(r*y1) + int(90*(1-k/200)) # (i // int(np.sqrt(len(nhl))))*20 + 20
    x_1 = int(r*x1) + int(90*(1-k/200)) # i % int(np.sqrt(len(nhl)))*20 + 20
    z_2 = z_1 + z2-z1
    y_2 = y_1 + y2-y1
    x_2 = x_1 + x2-x1
    ss2 = slice(z_1,z_2), slice(y_1,y_2), slice(x_1,x_2)
    return locals()

  # def recenter(newslices):
  #   max_extent = tuple(np.array([(d['z_2'], d['y_2'], d['x_2']) for d in newslices]).max(0))
  #   min_extent = tuple(np.array([(d['z_1'], d['y_1'], d['x_1']) for d in newslices]).min(0))

  k=200
  newslices = [g(n) for n in nhl]
  newshape = tuple(np.array([(d['z_2'], d['y_2'], d['x_2']) for d in newslices]).max(0)) + (3,)
  newimg = np.zeros(newshape)

  imgcounter = 0

  for k in range(0,200,10):
    imgcounter = k
    newimg[...] = 0
    newslices = [g(n) for n in nhl]
    for i in range(len(crops)):
      newimg[newslices[i]['ss2']] += crops[i]['crop']
    res = render_3chan(newimg)
    io.imsave('movie/test{:03d}.png'.format(imgcounter), res)

  T = spimagine.mat4_translate(0,0,-2)
  S = spimagine.mat4_scale(1,1,1)

  for r in np.r_[0:100,100:-1:-1]:
    imgcounter += 1
    R = spimagine.models.transform_model.mat4_rotation_euler(roll=r*np.pi/180)
    rend.set_modelView(T @ S @ R)
    res = render_3chan(newimg)
    io.imsave('movie/test{:03d}.png'.format(imgcounter), res)

  for r in np.r_[150:210]:
    imgcounter += 1
    R = spimagine.models.transform_model.mat4_rotation_euler(pitch=r*np.pi/180)
    rend.set_modelView(T @ S @ R)
    res = render_3chan(newimg)
    io.imsave('movie/test{:03d}.png'.format(imgcounter), res)

  for s in np.linspace(1,0.5,20):
    imgcounter += 1
    S = spimagine.mat4_scale(s,s,s)
    rend.set_modelView(T @ S @ R)
    res = render_3chan(newimg)
    io.imsave('movie/test{:03d}.png'.format(imgcounter), res)

def rotate_expanded(img,lab,dirname='rotate_expand'):

  rend = VolumeRenderer((500,500))
  rend.set_units([1.,1.,2])
  rend.set_projection(spimagine.mat4_perspective(70,1.,0.1,10))
  T = spimagine.mat4_translate(0,0,-2)
  S = spimagine.mat4_scale(1,1,1)

  rend.set_modelView(T @ S)
  rend.alphaPow = 0.3

  nhl = regionprops(lab)
  allimgs = []
  
  def f(n):
    z1,y1,x1,z2,y2,x2 = n.bbox
    ss1 = slice(z1,z2), slice(y1,y2), slice(x1,x2)
    crop = img[ss1].copy()
    crop[~n.filled_image] = 0
    crop = crop / np.percentile(crop,99.9,axis=(0,1,2),keepdims=True)
    return locals()

  crops = [f(n) for n in nhl]

  def g(n):
    z1,y1,x1,z2,y2,x2 = n.bbox
    r = 1 + k/200
    z_1 = int(r*z1) + int(90*(1-k/200)) # 100
    y_1 = int(r*y1) + int(90*(1-k/200)) # (i // int(np.sqrt(len(nhl))))*20 + 20
    x_1 = int(r*x1) + int(90*(1-k/200)) # i % int(np.sqrt(len(nhl)))*20 + 20
    z_2 = z_1 + z2-z1
    y_2 = y_1 + y2-y1
    x_2 = x_1 + x2-x1
    ss2 = slice(z_1,z_2), slice(y_1,y_2), slice(x_1,x_2)
    return locals()

  k=200
  newslices = [g(n) for n in nhl]
  newshape = tuple(np.array([(d['z_2'], d['y_2'], d['x_2']) for d in newslices]).max(0)) + (3,)
  newimg = np.zeros(newshape)

  imgcounter = 0

  for i in range(len(crops)):
    newimg[newslices[i]['ss2']] += crops[i]['crop']

  T = spimagine.mat4_translate(0,0,-2)
  S = spimagine.mat4_scale(1,1,1)

  mkdir(savedir / dirname)
  for r in np.r_[-30:30]:
    imgcounter += 1
    R = spimagine.models.transform_model.mat4_rotation_euler(pitch=r*np.pi/180)
    rend.set_modelView(T @ S @ R)
    res = render_3chan(rend, newimg)
    io.imsave(savedir / dirname / 'test{:03d}.png'.format(imgcounter), res)

def render_3chan(rend, img):
  def f(x):
    rend.set_data(x)
    err = rend.render()
    return rend.output
  res = np.stack([f(img[...,0]), f(img[...,1]), f(img[...,1])], -1)
  return res


# t1 = spimagine.TransformData(quatRot = spimagine.Quaternion(0.9684596728056643,-0.02435394587327544,0.24779909290609556,0.009399725862948783), zoom = 1.1566091783277792,
#                              dataPos = 0,
#                              minVal = 1e-06,
#                              maxVal = 6700.253663705347,
#                              gamma= 1.0448,
#                              translate = np.array([0, 0, 0]),
#                              bounds = np.array([-1.,  1., -1.,  1., -1.,  1.]),
#                              isBox = 2,
#                              isIso = False,
#                              alphaPow = 0.405)

# def update_rend_from_transformData(rend,td):
