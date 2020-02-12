import numpy as np
import torch
import matplotlib.pyplot as plt


## utils

def ft(arr):
  if type(arr) is np.ndarray:
    arr = torch.from_numpy(arr).float()
  elif type(arr) in [float,int]:
    arr = torch.Tensor([arr]).float()
  elif type(arr) in [list,tuple]:
    arr = torch.Tensor(arr).float()
  arr.requires_grad = True
  return arr

def ftng(arr):
  if type(arr) is np.ndarray:
    arr = torch.from_numpy(arr).float()
  elif type(arr) in [float,int]:
    arr = torch.Tensor([arr]).float()
  elif type(arr) in [list,tuple]:
    arr = torch.Tensor(arr).float()
  return arr

def dt(arr):
  if type(arr) is torch.Tensor:
    arr = arr.detach().numpy()
  # elif type(arr) in [float,int]:
  #   arr = torch.Tensor([arr]).float()
  elif type(arr) in [list,tuple]:
    arr = np.array([dt(x) for x in arr])
  return arr

def cdt(arr):
  return arr.clone().detach().numpy()

## fitting

def fit_gmm(img,pts3d,r0=30,n_iter=200):

  ## plot the histogram so we can see if 60 is a good p value
  plt.plot(np.percentile(img,np.linspace(0,100,100)))
  bg = np.percentile(img,60)

  # pts3d = pts(anno.man_track100)
  N = len(pts3d)
  # M = 400

  # background = ft([25]*N)
  radius = ft([[1,r0,r0]]*N)
  # height = ft([100]*N)
  height = ft([1]*N)
  params = []
  losses = []

  for i in range(n_iter):
    # M = max(int(radius.mean(0).prod(0)//20),40)
    # dom = np.random.rand(N,M,3)-0.5
    M = 3*9*9
    dom = ((np.indices((3,9,9)).T / (3-1,9-1,9-1)) - 0.5).reshape((-1,3))
    dom = (cdt(radius)[:,None]*dom[None])*2*1
    # dom = dom * (2*3,2*50,2*50).T
    # dom = dom * (5,20,20) ## sampling box width
    # dom = dom * cdt(radius[:,None])*2*2 ## sample from box of width 2x object radius
    dom_data  = (dom + pts3d[:,None,:]).clip(0,np.array(img.shape)-1).astype(int)
    data_vals = img[tuple(dom_data.reshape((-1,3)).T)].reshape((-1,M))
    data_vals = (data_vals-bg).clip(0).flatten() ## reduce weight of bg
    # data_vals = data_vals.flatten()
    # pred_vals = height[:,None] * torch.exp(-0.5*((ftng(dom)/radius[:,None])**2).sum(2))
    ## NOTE, in place operations are much faster than building a large array first, then summing.
    pred_vals = ftng(torch.zeros((M*N)))
    for j in range(N):
      pred_vals += height[j] * torch.exp(-0.5*((ftng(dom_data-pts3d[j])/radius[j])**2).sum(2)).flatten()
    loss = ((ftng(data_vals)-pred_vals)**2).mean() #+ 1e-3*((radius[:,0]-2.)**2).sum()
    loss.backward()
    ## differential learning rates, because different domains. radius lives in pixel domain, height in fluorescence...
    ## shouldn't the function take care of this by itself? 
    radius.data.sub_(radius.grad.data * .4) # 50 * .1)
    height.data.sub_(height.grad.data * .4) # .2 * .1)
    print(float(loss)); losses.append(loss)
    if i%1==0:
      params.append([radius.clone().detach().numpy(),height.clone().detach().numpy()])
    radius.data[:,0]=1
    radius.grad.data.zero_()
    height.grad.data.zero_()

  radius = radius.detach().numpy()
  height = height.detach().numpy()

  ## plot convergence
  plt.figure()
  radius_history = np.array([x[0] for x in params])
  plt.plot(radius_history[:,:,1])
  plt.plot(radius_history[:,:,2])
  plt.figure()
  height_history = np.array([x[1] for x in params])
  plt.plot(height_history)

  return radius,height,params,losses
