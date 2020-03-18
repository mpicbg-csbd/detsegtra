import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchsummary import summary

@DeprecationWarning
def conv_many(*cs):
  N = len(cs)-1
  convs = [nn.Conv3d(cs[i],cs[i+1],(3,5,5),padding=(1,2,2)) for i in range(N)]
  relus = [nn.ReLU() for i in range(N)]
  res = [0]*N*2
  res[::2] = convs
  res[1::2] = relus
  return nn.Sequential(*res)


def receptivefield(net,kern=(3,5,5)):
  """
  calculate receptive field / receptive kernel of Conv net
  WARNING: overwrites net weights. save weights first!
  WARNING: only works with relu activations, pooling and upsampling.
  """
  def rfweights(m):
    if type(m) in [nn.Conv3d,nn.Conv2d]:
      m.weight.data.fill_(1/np.prod(kern)) ## conv kernel 3*5*5
      m.bias.data.fill_(0.0)
  net.apply(rfweights);
  if len(kern)==3:
    x0 = np.zeros((128,128,128)); x0[64,64,64]=1;
  elif len(kern)==2:
    x0 = np.zeros((256,256)); x0[128,128]=1;
  xout = net.cuda()(torch.from_numpy(x0)[None,None].float().cuda()).detach().cpu().numpy()
  return xout

def init_weights(net):
  def f(m):
    if type(m) == nn.Conv3d:
      torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
      m.bias.data.fill_(0.05)
  net.apply(f);

def test_weights(net):
  table = ["weight mean,weight std,bias mean,bias std".split(',')]
  def f(m):
    if type(m) == nn.Conv3d:
      table.append([float(m.weight.data.mean()),float(m.weight.data.std()),float(m.bias.data.mean()),float(m.bias.data.std())])
      print(m)
  net.apply(f)
  return table

## 3d nets

def conv2(c0,c1,c2,kernsize, padding):
  return nn.Sequential(
    nn.Conv3d(c0,c1,kernsize, padding=padding),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv3d(c1,c2,kernsize, padding=padding),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Unet2(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2, self).__init__()

    self.l_ab = conv2(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 4*c, 2*c)
    self.l_gh = conv2(4*c, 2*c, 1*c)
    self.l_ij = conv2(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def load_old_state(self, state):
    self.load_state_dict(state,strict=False) ## loads most things correctly, but now we have to fix the missing keys
    self.l_ef[0].weight.data[...] = state['l_e.0.weight'].data
    self.l_ef[0].bias.data[...]   = state['l_e.0.bias'].data
    self.l_ef[2].weight.data[...] = state['l_f.0.weight'].data
    self.l_ef[2].bias.data[...]   = state['l_f.0.bias'].data
    self.l_gh[0].weight.data[...] = state['l_g.0.weight'].data
    self.l_gh[0].bias.data[...]   = state['l_g.0.bias'].data
    self.l_gh[2].weight.data[...] = state['l_h.0.weight'].data
    self.l_gh[2].bias.data[...]   = state['l_h.0.bias'].data

  def forward(self, x):

    c1 = self.l_ab(x)
    c2 = nn.MaxPool3d((1,2,2))(c1)
    c2 = self.l_cd(c2)
    c3 = nn.MaxPool3d((1,2,2))(c2)
    c3 = self.l_ef(c3)

    c3 = F.interpolate(c3,scale_factor=(1,2,2))
    c3 = torch.cat([c3,c2],1)
    c3 = self.l_gh(c3)
    c3 = F.interpolate(c3,scale_factor=(1,2,2))
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ij(c3)
    out1 = self.l_k(c3)

    return out1

class Unet3(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet3, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer

    self.l_ab = conv2(io[0][0], c, c, kernsize, padding=pool)
    self.l_cd = conv2(1*c, 2*c,  2*c, kernsize, padding=pool)
    self.l_ef = conv2(2*c, 4*c,  4*c, kernsize, padding=pool)
    self.l_gh = conv2(4*c, 8*c,  4*c, kernsize, padding=pool)
    self.l_ij = conv2(8*c, 4*c,  2*c, kernsize, padding=pool)
    self.l_kl = conv2(4*c, 2*c,  1*c, kernsize, padding=pool)
    self.l_mn = conv2(2*c, 1*c,  1*c, kernsize, padding=pool)
    
    self.l_o  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def forward(self, x):

    c1 = self.l_ab(x)
    c2 = nn.MaxPool3d(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = nn.MaxPool3d(self.pool)(c2)
    c3 = self.l_ef(c3)
    c4 = nn.MaxPool3d(self.pool)(c3)
    c4 = self.l_gh(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c3],1)
    c4 = self.l_ij(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c2],1)
    c4 = self.l_kl(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c1],1)
    c4 = self.l_mn(c4)
    out1 = self.l_o(c4)

    return out1

## 2d nets

def conv2_2d(c0,c1,c2):
  return nn.Sequential(
    nn.Conv2d(c0,c1,(5,5),padding=(2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv2d(c1,c2,(5,5),padding=(2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Unet2_2d(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2_2d, self).__init__()

    self.l_ab = conv2_2d(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2_2d(1*c, 2*c, 2*c)
    self.l_ef = conv2_2d(2*c, 4*c, 2*c)
    self.l_gh = conv2_2d(4*c, 2*c, 1*c)
    self.l_ij = conv2_2d(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv2d(1*c,io[1][0],(1,1),padding=0), finallayer())

  # def load_old_state(self, state):
  #   self.load_state_dict(state,strict=False) ## loads most things correctly, but now we have to fix the missing keys
  #   self.l_ef[0].weight.data[...] = state['l_e.0.weight'].data
  #   self.l_ef[0].bias.data[...]   = state['l_e.0.bias'].data
  #   self.l_ef[2].weight.data[...] = state['l_f.0.weight'].data
  #   self.l_ef[2].bias.data[...]   = state['l_f.0.bias'].data
  #   self.l_gh[0].weight.data[...] = state['l_g.0.weight'].data
  #   self.l_gh[0].bias.data[...]   = state['l_g.0.bias'].data
  #   self.l_gh[2].weight.data[...] = state['l_h.0.weight'].data
  #   self.l_gh[2].bias.data[...]   = state['l_h.0.bias'].data

  def forward(self, x):

    c1 = self.l_ab(x)
    c2 = nn.MaxPool2d((2,2))(c1)
    c2 = self.l_cd(c2)
    c3 = nn.MaxPool2d((2,2))(c2)
    c3 = self.l_ef(c3)  ## bottom, central layer

    c3 = F.interpolate(c3,scale_factor=(2,2))
    c3 = torch.cat([c3,c2],1)
    c3 = self.l_gh(c3)
    c3 = F.interpolate(c3,scale_factor=(2,2))
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ij(c3)
    out1 = self.l_k(c3)

    return out1

## resnets are incomplete

def conv_res(c0,c1,c2):
  return nn.Sequential(
    nn.Conv3d(c0,c1,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv3d(c1,c2,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    # nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Res1(nn.Module):
  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2, self).__init__()

    self.l_ab = conv2(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 1*c, 1*c)
    # self.l_gh = conv2(4*c, 2*c, 1*c)
    # self.l_ij = conv2(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def forward(self, x):

    c1 = nn.Relu()(self._lab(x)  + x)
    c2 = nn.Relu()(self._lcd(c1) + c1)
    c3 = nn.Relu()(self._lef(c2) + c2)
    out1 = self.l_k(c3)

    return out1


## utils

def pretty_size(size):
  """Pretty prints a torch.Size object"""
  assert(isinstance(size, torch.Size))
  return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
  """Prints a list of the Tensors being tracked by the garbage collector."""
  import gc
  total_size = 0
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj):
        if not gpu_only or obj.is_cuda:
          print("%s:%s%s %s" % (type(obj).__name__, 
                      " GPU" if obj.is_cuda else "",
                      " pinned" if obj.is_pinned else "",
                      pretty_size(obj.size())))
          total_size += obj.numel()
      elif hasattr(obj, "data") and torch.is_tensor(obj.data):
        if not gpu_only or obj.is_cuda:
          print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                           type(obj.data).__name__, 
                           " GPU" if obj.is_cuda else "",
                           " pinned" if obj.data.is_pinned else "",
                           " grad" if obj.requires_grad else "", 
                           " volatile" if obj.volatile else "",
                           pretty_size(obj.data.size())))
          total_size += obj.data.numel()
    except Exception as e:
      pass        
  print("Total size:", total_size)

## prediction with tiling

import numpy as np
import itertools
from math import ceil

def apply_net_tiled_3d(net,img):
  """
  Does not perform normalization.
  Applies net to image with dims Channels,Z,Y,X.
  Assume 3x or less max pooling layers => (U-net) translational symmetry with period 2^n for n in [0,1,2,3].
  """

  # borders           = [8,24,24] ## border width within each patch that is thrown away after prediction
  # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
  # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
  # stride            = [16,200,200] ## same as patchshape in this case
  # def g(n,m): return floor(n/m)*m-n ## f(n,m) gives un-padding needed for n to be divisible by m
  def f(n,m): return ceil(n/m)*m-n ## gives padding needed for n to be divisible by m

  a,b,c = img.shape[1:]

  ## extra border needed for stride % 8 = 0. read as e.g. "ImagePad_Z"
  ip_z,ip_y,ip_x = f(a,8),f(b,8),f(c,8) 
  
  # pp_z,pp_y,pp_x = 8,32,32
  # DZ,DY,DX = 16,200,200

  ## max total size with Unet3 16 input channels (64,528,528) = 
  ## padding per patch. must be divisible by 8. read as e.g. "PatchPad_Z"
  pp_z,pp_y,pp_x = 8,64,64
  ## inner patch size (does not include patch border. also will be smaller at boundary)
  DZ,DY,DX = 48,400,400

  img_padded = np.pad(img,[(0,0),(pp_z,pp_z+ip_z),(pp_y,pp_y+ip_y),(pp_x,pp_x+ip_x)],mode='constant')
  output = np.zeros(img.shape)

  ## start coordinates for each patch in the padded input. each stride must be divisible by 8.
  zs = np.r_[:a:DZ]
  ys = np.r_[:b:DY]
  xs = np.r_[:c:DX]

  ## start coordinates of padded input (e.g. top left patch corner)
  for x,y,z in itertools.product(xs,ys,zs):
    ## end coordinates of padded input (including patch border)
    ze,ye,xe = min(z+DZ,a+ip_z) + 2*pp_z, min(y+DY,b+ip_y) + 2*pp_y, min(x+DX,c+ip_x) + 2*pp_x
    patch = img_padded[:,z:ze,y:ye,x:xe]
    with torch.no_grad():
      patch = torch.from_numpy(patch).cuda().float()
      patch = net(patch[None])[0,:,pp_z:-pp_z,pp_y:-pp_y,pp_x:-pp_x].detach().cpu().numpy()
    ## end coordinates of unpadded output (not including border)
    ae,be,ce = min(z+DZ,a),min(y+DY,b),min(x+DX,c)
    ## use "ae-z" because patch size changes near boundary
    output[:,z:ae,y:be,x:ce] = patch[:,:ae-z,:be-y,:ce-x]

  return output
