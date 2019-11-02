import subprocess
import pickle
import numpy as np
import spimagine
from spimagine.volumerender.volumerender import VolumeRenderer
from numpy_utils import normalize2


def sync(flip=False):
  exc = ['*.npy', '*.net', '*.tif', '*.npz', '*.h5', '*.pkl']
  sharedext = "devseg_data/"
  d1 = "efal:/projects/project-broaddus/"  + sharedext
  d2 = "/Users/broaddus/Desktop/Projects/" + sharedext
  if flip: d1,d2 = d2,d1
  exclude = ''.join([' --exclude="{x}"'.format(x=x) for x in exc])
  cmd = "rsync -arv {exclude} {fromdir} {todir}".format(exclude=exclude,fromdir=d1,todir=d2)
  res = subprocess.run([cmd], shell=True)
  print(res)

def sync_all(sharedext = "devseg_data/raw/zfish/data/img001.tif", way='U'):
  exc = []
  d1 = "efal:/projects/project-broaddus/"  + sharedext
  d2 = "/Users/broaddus/Desktop/Projects/" + sharedext
  
  if way.upper() == 'D':
    exclude = ''.join([' --exclude="{x}"'.format(x=x) for x in exc])
    cmd = "rsync -arv {exclude} {fromdir} {todir}".format(exclude=exclude,fromdir=d1,todir=d2)
    res = subprocess.run([cmd], shell=True)
    print(res)

  if way.upper() == 'U':
    d1,d2 = d2,d1
    exclude = ''.join([' --exclude="{x}"'.format(x=x) for x in exc])
    cmd = "rsync -arv {exclude} {fromdir} {todir}".format(exclude=exclude,fromdir=d1,todir=d2)
    res = subprocess.run([cmd], shell=True)
    print(res)


def qloade():
  res = subprocess.run(['rsync efal:qsave.npy .'], shell=True)
  print(res)
  return np.load('qsave.npy')

def qload(projectdir="/projects/project-broaddus/devseg_code/detect/"):
  # import subprocess
  res = subprocess.run(['rsync broaddus@falcon:{}qsave.npy .'.format(projectdir)], shell=True)
  print(res)
  return np.load('qsave.npy')

def ploade():
  res = subprocess.run(['rsync efal:psave.pkl .'], shell=True)
  print(res)
  return pickle.load(open('psave.pkl','rb'))

def pload():
  res = subprocess.run(['rsync broaddus@falcon:psave.pkl .'], shell=True)
  print(res)
  return pickle.load(open('psave.pkl','rb'))

def mkrgb(img):
  img = img[...,None]
  img = img[...,[0,0,0]]
  return img

def rotategif(img):
  from array2gif import write_gif
  from scipy.ndimage import affine_transform

  img  = np.pad(img,[(0,144),(20,20),(0,0)], mode='constant')
  imglist = []
  for i in range(3,21,3):
    count += 1
    c,s = np.cos(np.pi/180*i), np.sin(np.pi/180*i)
    rot = rot = np.array([[c,-s,0,],[s,c,0],[0,0,1]])
    newimg = affine_transform(img,rot)
    newimg = newimg.max(0)
    imglist.append(newimg)

  imglist.append(img.max(0))
  imglist = np.array(imglist)
  imglist = imglist/imglist.max()*255
  imglist = mkrgb(imglist)
  write_gif(imglist, 'testgif2.gif', fps=10)

def onealledges(img,a=0.5):
  img[0 , 0, :] = a
  img[0 ,-1, :] = a
  img[-1, 0, :] = a
  img[-1,-1, :] = a
  img[0 , :, 0] = a
  img[0 , :,-1] = a
  img[-1, :, 0] = a
  img[-1, :,-1] = a
  img[ :,0 , 0] = a
  img[ :,0 ,-1] = a
  img[ :,-1, 0] = a
  img[ :,-1,-1] = a
  return img

def render_3chan(img):
  res = np.stack([volumerendernumpy(img[:,0]), volumerendernumpy(img[:,1]), volumerendernumpy(img[:,1])], -1)
  return res


t1 = spimagine.TransformData(quatRot = spimagine.Quaternion(0.9684596728056643,-0.02435394587327544,0.24779909290609556,0.009399725862948783), zoom = 1.1566091783277792,
                             dataPos = 0,
                             minVal = 1e-06,
                             maxVal = 6700.253663705347,
                             gamma= 1.0448,
                             translate = np.array([0, 0, 0]),
                             bounds = np.array([-1.,  1., -1.,  1., -1.,  1.]),
                             isBox = 2,
                             isIso = False,
                             alphaPow = 0.405)

# def update_rend_from_transformData(rend,td):

def volumerendernumpy(img):
  img = np.pad(img,[(20,20),(20,20),(20,20)],mode='constant')
  img = normalize2(img, 2,99.5,(0,1,2))
  # img = onealledges(img)
  rend = VolumeRenderer((500,500))
  rend.set_data(img)
  m = spimagine.models.transform_model.mat4_rotation_euler(yaw=5*np.pi/180)
  rend.set_modelView(m)
  # rend.alphaPow = 0.5
  rend.set_units([1.,1.,5])
  rend.set_projection(spimagine.mat4_perspective(70,1.,0.1,10))
  rend.set_modelView(np.dot(spimagine.mat4_translate(0,0,-2),spimagine.mat4_scale(1,1,1)))
  err = rend.render()
  return rend.output

