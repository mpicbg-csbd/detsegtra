doc="""
UNET architecture for pixelwise classification
"""

import numpy as np
import skimage.io as io
import json
import random
from pathlib import Path

from .defaults.ipython import perm
from . import nhl_tools

from keras.activations import softmax
from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import Input, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, core, Dropout
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator




def add_z_to_chan(img, dz, ind=None, axes="ZCYX"):
  assert img.ndim == 4

  ## by default do the entire stack
  if ind is None:
    ind = np.arange(img.shape[0])

  ## pad img
  img = perm(img, axes, "ZCYX")
  pad = [(dz,dz)] + [(0,0)]*3
  img = np.pad(img, pad, 'reflect')

  ## allow single ind
  if not hasattr(ind, "__len__"):
    ind = [ind]

  def add_single(i):
    res = img[i-dz:i+dz+1]
    a,b,c,d = res.shape
    res = res.reshape((a*b,c,d))
    res = perm(res, "CYX", "YXC")
    return res

  ind = np.array(ind) + dz
  res = np.stack([add_single(i) for i in ind], axis=0)
  res = perm(res, "ZYXC", axes)

  return res

def weighted_categorical_crossentropy(classweights=(1., 1.), itd=1, BEnd=K):
    """
    last channel of y_pred gives pixelwise weights
    """
    classweights = np.array(classweights)
    mean = BEnd.mean
    log  = BEnd.log
    summ = BEnd.sum
    eps  = K.epsilon()
    if itd==0:
        ss = [slice(None), slice(None), slice(None), slice(None)]
    else:
        ss = [slice(None), slice(itd,-itd), slice(itd,-itd), slice(None)]
    def catcross(y_true, y_pred):
        yt = y_true[ss]
        yp = y_pred[ss]
        ws = yt[...,-1]
        yt = yt[...,:-1]
        ce = ws[...,np.newaxis] * yt * log(yp + eps)
        ce = mean(ce, axis=(0,1,2))
        result = classweights * ce
        result = -summ(result)
        return result
    return catcross

def my_categorical_crossentropy(classweights=(1., 1.), itd=1, BEnd=K):
    """
    NOTE: The default classweights assumes 2 classes, but the loss works for arbitrary classes if we simply change the length of the classweights arg.
    
    Also, we can replace K with numpy to get a function we can actually evaluate (not just pass to compile)!
    """
    classweights = np.array(classweights)
    mean = BEnd.mean
    log  = BEnd.log
    summ = BEnd.sum
    eps  = K.epsilon()
    if itd==0:
        ss = [slice(None), slice(None), slice(None), slice(None)]
    else:
        ss = [slice(None), slice(itd,-itd), slice(itd,-itd), slice(None)]
    def catcross(y_true, y_pred):
        yt = y_true[ss]
        yp = y_pred[ss]
        ce = yt * log(yp + eps)
        ce = mean(ce, axis=(0,1,2))
        result = classweights * ce
        result = -summ(result)
        return result
    return catcross



def get_unet_n_pool(n_pool=2, inputchan=1, n_classes=2, n_convolutions_first_layer=32, 
                    dropout_fraction=0.2, last_activation='softmax', kern_width=3, ndim=2):
    """
    The info travel distance is given by info_travel_dist(n_pool, kern_width)
    """

    if K.image_dim_ordering() == 'th':
      if ndim==2:
        inputs = Input((inputchan, None, None))
      elif ndim==3:
        inputs = Input((inputchan, None, None, None))
      concatax = 1
      chan = 'channels_first'
    elif K.image_dim_ordering() == 'tf':
      if ndim==2:
        inputs = Input((None, None, inputchan))
      elif ndim==3:
        inputs = Input((None, None, None, inputchan))
      concatax = 3 + ndim - 2
      chan = 'channels_last'

    if ndim==2:
        Convnd  = Conv2D
        Poolnd  = MaxPooling2D
        Upcatnd = UpSampling2D
        convsize = (kern_width, kern_width)
        poolsize = (2,2)
        upsampsize = (2,2)
        finalconvsize = (1, 1)
        permdims = (2,3,1)
    elif ndim==3:
        Convnd  = Conv3D
        Poolnd  = MaxPooling3D
        Upcatnd = UpSampling3D
        convsize = (kern_width, kern_width, kern_width)
        poolsize = (2,2,2)
        upsampsize = (2,2,2)
        finalconvsize = (1, 1, 1)
        permdims = (2,3,4,1)

    def Conv(w):
        return Convnd(w, convsize, padding='same', data_format=chan, activation='relu', kernel_initializer='he_normal')
    def Pool():
        return Poolnd(pool_size=poolsize, data_format=chan)
    def Upsa():
        return Upcatnd(size=upsampsize, data_format=chan)
    
    d = dropout_fraction
    
    def cdcp(s, inpt):
        """
        Conv, Drop, Conv, Pool
        """
        conv = Conv(s)(inpt)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        pool = Pool()(conv)
        return conv, pool

    def uacdc(s, inpt, skip):
        """
        Up, cAt, Conv, Drop, Conv
        """
        up   = Upsa()(inpt)
        cat  = Concatenate(axis=concatax)([up, skip])
        conv = Conv(s)(cat)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        return conv

    # once we've defined the above terms, the entire unet just takes a few lines ;)

    # holds the output of convolutions on the contracting path
    conv_layers = []

    # the first conv comes from the inputs
    s = n_convolutions_first_layer
    conv, pool = cdcp(s, inputs)
    conv_layers.append(conv)

    # then the recursively describeable contracting part
    for _ in range(n_pool-1):
        s *= 2
        conv, pool = cdcp(s, pool)
        conv_layers.append(conv)

    # the flat bottom. no max pooling.
    s *= 2
    conv_bottom = Conv(s)(pool)
    conv_bottom = Dropout(d)(conv_bottom)
    conv_bottom = Conv(s)(conv_bottom)
    
    # now each time we cut s in half and build the next UACDC
    s = s//2
    up = uacdc(s, conv_bottom, conv_layers[-1])

    # recursively describeable expanding path
    for conv in reversed(conv_layers[:-1]):
        s = s//2
        up = uacdc(s, up, conv)

    # final (1,1) convolutions and activation
    acti_layer = Convnd(n_classes, finalconvsize, padding='same', data_format=chan, activation=None)(up)
    if K.image_dim_ordering() == 'th':
        acti_layer = core.Permute(permdims)(acti_layer)
    acti_layer = core.Activation(last_activation)(acti_layer)
    model = Model(inputs=inputs, outputs=acti_layer)
    return model

def info_travel_dist(n_maxpool, conv=3):
    """
    n_maxpool: number of 2x downsampling steps
    conv: the width of the convolution kernel (e.g. "3" for standard 3x3 kernel.)
    returns: the info travel distance == the amount of width that is lost in a patch / 2
        i.e. the distance from the pixel at the center of a maxpool grid to the edge of
        the grid.
    """
    conv2 = 2*(conv-1)
    width = 0
    for i in range(n_maxpool):
        width -= conv2
        width /= 2
    width -= conv2
    for i in range(n_maxpool):
        width *= 2
        width -= conv2
    return int(-width/2)

def batch_generator_patches(X, Y, train_params, verbose=False):
    epoch = 0
    tp = train_params
    while (True):
        epoch += 1
        current_idx = 0
        batchnum = 0
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        Y = Y[inds]
        while batchnum < tp['batches_per_epoch']:
            Xbatch, Ybatch = X[current_idx:current_idx + tp['batch_size']].copy(), Y[current_idx:current_idx + tp['batch_size']].copy()
            # io.imsave('X.tif', Xbatch, plugin='tifffile')
            # io.imsave('Y.tif', Ybatch, plugin='tifffile')

            current_idx += tp['batch_size']

            for i in range(Xbatch.shape[0]):
                x = Xbatch[i]
                y = Ybatch[i]
                x,y = warping.randomly_augment_patches(x, y, tp['noise'], tp['flipLR'], tp['warping_size'], tp['rotate_angle_max'])
                Xbatch[i] = x
                Xbatch = normalize_X(Xbatch)
                Ybatch[i] = y

            # io.imsave('Xauged.tif', Xbatch.astype('float32'), plugin='tifffile')
            # io.imsave('Yauged.tif', Ybatch.astype('float32'), plugin='tifffile')

            Xbatch = add_singleton_dim(Xbatch)
            Ybatch = labels_to_activations(Ybatch, tp['n_classes'])

            # print('xshape', Xbatch.shape)
            # print('yshape', Ybatch.shape)

            batchnum += 1
            yield Xbatch, Ybatch

def batch_generator_patches_aug(X, Y, 
                                # steps_per_epoch=100,
                                batch_size=4,
                                augment_and_norm=lambda x,y:(x,y),
                                verbose=False,
                                savepath=None):

    if type(savepath) is str:
        savepath = Path(savepath)

    epoch = 0
    while (True):
        epoch += 1
        current_idx = 0
        batchnum = 0
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        Y = Y[inds]
        # while batchnum < steps_per_epoch:

        Xepoch = []
        Yepoch = []
        
        while current_idx < X.shape[0]:
            i0 = current_idx
            i1 = min(current_idx + batch_size, X.shape[0])
            Xbatch, Ybatch = X[i0:i1].copy(), Y[i0:i1].copy()
            # io.imsave('X.tif', Xbatch, plugin='tifffile')
            # io.imsave('Y.tif', Ybatch, plugin='tifffile')

            current_idx += batch_size
            batchnum += 1

            for i in range(Xbatch.shape[0]):
                x = Xbatch[i]
                y = Ybatch[i]
                x,y = augment_and_norm(x, y)
                Xbatch[i] = x
                Ybatch[i] = y

            if epoch==1 and savepath is not None:
                Xepoch.append(Xbatch)
                Yepoch.append(Ybatch)

            # io.imsave('Xauged.tif', Xbatch.astype('float32'), plugin='tifffile')
            # io.imsave('Yauged.tif', Ybatch.astype('float32'), plugin='tifffile')

            # print('xshape', Xbatch.shape)
            # print('yshape', Ybatch.shape)

            yield Xbatch, Ybatch

        if epoch==1 and savepath is not None:
            np.savez(savepath / 'XY_train', x=np.array(Xepoch), y=np.array(Yepoch))

def batch_generator_pred_zchannel(X,
                        # steps_per_epoch=100,
                        axes='ZCYX',
                        batch_size=4,
                        dz=0,
                        patch_apply=lambda x,y:(x,y),
                        verbose=False,
                        savepath=None):

    "no shuffle. no epochs. no augment, but still apply arbitrary funcs."

    batchnum = 0

    zmax = X.shape[0]+dz
    pad = [(dz,dz)] + [(0,0)]*3
    X = np.pad(X, pad, 'reflect')
    current_idx = dz
    
    while current_idx < zmax:
        i0 = current_idx
        i1 = min(current_idx + batch_size, zmax)
        print(i0,i1)
        Xbatch = np.stack([ll.add_z_to_chan(X,i,dz) for i in range(i0,i1)])

        current_idx += batch_size
        batchnum += 1

        yield Xbatch
