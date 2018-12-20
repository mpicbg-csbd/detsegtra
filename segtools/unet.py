
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
from keras.layers import Convolution2D, BatchNormalization, Activation
from keras.layers import Input, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, core, Dropout, AveragePooling2D, AveragePooling3D
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


eps = K.epsilon()

def crossentropy_w(yt,yp):
    ws = yt[...,-1]
    ws = ws[...,np.newaxis]
    yt = yt[...,:-1]
    ce = ws * yt * K.log(yp + eps)
    ce = -K.mean(ce)
    return ce

def get_unet_n_pool(input0, 
                    n_pool=2, 
                    n_convolutions_first_layer=32,
                    dropout_fraction=0.2, 
                    kern_width=3):
    """
    The info travel distance is given by info_travel_dist(n_pool, kern_width)
    """
    ndim = len(input0.shape)-2

    if K.image_dim_ordering() == 'th':
      concatax = 1
      chan = 'channels_first'
    elif K.image_dim_ordering() == 'tf':
      concatax = 3 + ndim - 2
      chan = 'channels_last'

    if ndim==2:
        Convnd  = Conv2D
        Poolnd  = MaxPooling2D
        Upcatnd = UpSampling2D
        convsize = (kern_width, kern_width)
        poolsize = (2,2)
        upsampsize = (2,2)
    elif ndim==3:
        Convnd  = Conv3D
        Poolnd  = MaxPooling3D
        Upcatnd = UpSampling3D
        convsize = (kern_width, kern_width, kern_width)
        poolsize = (2,2,2)
        upsampsize = (2,2,2)

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
    conv, pool = cdcp(s, input0)
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

    return up

def get_unet_n_poolv2(input0,
                    n_pool=2,
                    n_convolutions_first_layer=32,
                    dropout_fraction=0.2,
                    convs_per_layer=2,
                    convsize=(3,3,3),
                    poolsize=(2,2,2),
                    upsampsize=(2,2,2),
                    activation='relu',
                    batch_norm=True,
                    n_conv_per_scale=2,
                    ):
    """
    The info travel distance is given by info_travel_dist(n_pool, kern_width)
    """
    ndim = len(convsize)

    if K.image_dim_ordering() == 'th':
      concatax = 1
      chan = 'channels_first'
    elif K.image_dim_ordering() == 'tf':
      concatax = 3 + ndim - 2
      chan = 'channels_last'

    if ndim==2:
        Convnd  = Conv2D
        Poolnd  = MaxPooling2D
        Upcatnd = UpSampling2D
    elif ndim==3:
        Convnd  = Conv3D
        Poolnd  = MaxPooling3D
        Upcatnd = UpSampling3D

    def Conv(n_features):
        def f(layer):
            l = Convnd(n_features, convsize, padding='same', data_format=chan, kernel_initializer='he_normal')(layer)
            if batch_norm: l = BatchNormalization()(l)
            if activation: l = Activation(activation)(l)
            l = Dropout(dropout_fraction)(l)
            return l
        return f
    def Pool():
        return Poolnd(pool_size=poolsize, data_format=chan)
    def Upsa():
        return Upcatnd(size=upsampsize, data_format=chan)
    
    def convs_and_pool(s, inpt):
        """
        Conv,Conv,...,etc,Pool
        """
        conv = inpt
        for _ in range(n_conv_per_scale):
            conv = Conv(s)(conv)
        pool = Pool()(conv)
        return conv, pool

    def upsample_and_convs(s, inpt, skip):
        """
        Up, cAt, Conv, Conv, ..., etc
        """
        up   = Upsa()(inpt)
        cat  = Concatenate(axis=concatax)([up, skip])
        for _ in range(n_conv_per_scale):
            cat = Conv(s)(cat)
        return cat

    # holds the output of convolutions on the contracting path
    # the first conv comes from the inputs
    skip_layers = []
    n_features = n_convolutions_first_layer
    pool = input0

    # then the recursively describeable contracting part
    for _ in range(n_pool):
        conv, pool = convs_and_pool(n_features, pool)
        skip_layers.append(conv)
        n_features *= 2

    # the flat bottom. no max pooling.
    for _ in range(n_conv_per_scale-1):
        pool = Conv(n_features)(pool)
    
    # now each time we cut n_features in half
    up = pool
    # recursively describeable expanding path
    for layer in reversed(skip_layers):
        up = upsample_and_convs(n_features, up, layer)
        n_features = n_features//2

    return up


def get_unet_n_pool_recep(input0, n_pool=2, n_convolutions_first_layer=32,
                    dropout_fraction=0.2, kern_width=3):
    """
    The info travel distance is given by info_travel_dist(n_pool, kern_width)
    """
    ndim = len(input0.shape)-2

    if K.image_dim_ordering() == 'th':
      concatax = 1
      chan = 'channels_first'
    elif K.image_dim_ordering() == 'tf':
      concatax = 3 + ndim - 2
      chan = 'channels_last'

    if ndim==2:
        Convnd  = Conv2D
        Poolnd  = AveragePooling2D
        Upcatnd = UpSampling2D
        convsize = (kern_width, kern_width)
        poolsize = (2,2)
        upsampsize = (2,2)
    elif ndim==3:
        Convnd  = Conv3D
        Poolnd  = AveragePooling3D
        Upcatnd = UpSampling3D
        convsize = (kern_width, kern_width, kern_width)
        poolsize = (2,2,2)
        upsampsize = (2,2,2)

    def Conv(w):
        return Convnd(w, convsize, padding='same', data_format=chan, activation='relu', kernel_initializer='ones', bias_initializer='zeros')
    def Pool():
        return Poolnd(pool_size=poolsize, data_format=chan)
    def Upsa():
        return Upcatnd(size=upsampsize, data_format=chan)
    
    d = dropout_fraction
    
    def cdcp(s, inpt):
        """
        Conv, Drop, Conv, Pool
        """
        s=1
        conv = Conv(s)(inpt)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        pool = Pool()(conv)
        return conv, pool

    def uacdc(s, inpt, skip):
        """
        Up, cAt, Conv, Drop, Conv
        """
        s=1
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
    conv, pool = cdcp(s, input0)
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

    return up

def acti(input0, n_classes, last_activation='softmax', **kwargs):
    "final (1,1) convolutions and activation"
    ndim = len(input0.shape)-2
    if ndim==2:
        Convnd  = Conv2D
        finalconvsize = (1,1)
    elif ndim==3:
        Convnd  = Conv3D
        finalconvsize = (1,1,1)
    acti_layer = Convnd(n_classes, finalconvsize, padding='same', activation=None, kernel_initializer='ones', bias_initializer='zeros')(input0)
    acti_layer = core.Activation(last_activation, **kwargs)(acti_layer)
    return acti_layer

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


## all deprected?

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
                                net=None,
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
        # X = Xin[inds]
        # Y = Yin[inds]

        Xepoch = []
        Yepoch = []

        if net is not None:
            pred_xs = net.predict(X[[0,10,50]], batch_size=1)
        
        while current_idx < X.shape[0]:
            i0 = current_idx
            i1 = min(current_idx + batch_size, X.shape[0])
            Xbatch, Ybatch = X[inds][i0:i1].copy(), Y[inds][i0:i1].copy()
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
                if Xbatch.shape[0]==batch_size: ## ignore last frame
                    Xepoch.append(Xbatch)
                    Yepoch.append(Ybatch)


            # io.imsave('Xauged.tif', Xbatch.astype('float32'), plugin='tifffile')
            # io.imsave('Yauged.tif', Ybatch.astype('float32'), plugin='tifffile')

            # print('xshape', Xbatch.shape)
            # print('yshape', Ybatch.shape)

            yield Xbatch, Ybatch

        if epoch==1 and savepath is not None:
            x = np.array(Xepoch)
            y = np.array(Yepoch)
            print("Saving first epoch w shape: ", x.shape)
            print("Saving first epoch w shape: ", y.shape)
            np.savez(str(savepath / 'XY_train'), x=x, y=y)

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
