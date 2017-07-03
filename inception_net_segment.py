from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import  merge, Convolution2D, UpSampling2D
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.initializers import Constant
from keras.layers.pooling import MaxPooling2D
from keras import layers
from metric import dice_coef, dice_coef_loss

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 192
img_cols = 208




def residual_block(inputs,depth, activation='relu'):
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    residual = Conv2D(depth,(1,1),kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
    residual = BatchNormalization()(residual)
    residual = actv()(residual)
    residual = Conv2D(depth,(1,1),kernel_initializer='he_normal', padding='same', data_format='channels_last')(residual)
    residual = BatchNormalization()(residual)
    return layers.add([inputs,residual])
    

# TODO: need to experiment with different depths
def inception_block(inputs, depth, factor, batch_mode=0, splitted=False, activation='relu',):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    
    # convolution2D function is deprecated
    c1_1 = Conv2D(depth//4, (1,1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
    
    c2_1 = Conv2D(depth//(16//factor), (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
    c2_1 = BatchNormalization()(c2_1)
    c2_1 = actv()(c2_1)
    
    if splitted:
        c2_2 = Conv2D(depth//4, (1, 3), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c2_1)
        c2_2 = BatchNormalization()(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(depth//4, (3, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c2_2)
    else:
        c2_3 = Conv2D(depth//4, (3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c2_1)
    
    c3_1 = Conv2D(depth//(16//factor), (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
    c3_1 = BatchNormalization()(c3_1)
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Conv2D(depth//4, (1, 5), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_1)
        c3_2 = BatchNormalization()(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(depth//4, (5, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_2)
    else:
        c3_3 = Conv2D(depth//4, (5, 5), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_1)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_last')(inputs)
    c4_2 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(p4_1)
    
    res = concatenate([c1_1, c2_3, c3_3, c4_2])
    res = BatchNormalization()(res)
    res = actv()(res)
    return res

# inception with assymetrically factorized 7X7 comvolutions
# used only when the grid size is between 12 and 20
def assymetrical_inception(inputs, depth, kernel_size, batch_mode=0, splitted=False, activation='relu'):
   actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
   
   c1_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)

   pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_last')(inputs)
   c2_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(pool)
   
   c3_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
   c3_1 = BatchNormalization()(c3_1)
   c3_1 = actv()(c3_1)
   c3_2 = Conv2D(depth//4, (1, kernel_size), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_1)
   c3_2 = BatchNormalization()(c3_2)
   c3_2 = actv()(c3_2)
   c3_3 = Conv2D(depth//4, (kernel_size, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_2)

   c4_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
   c4_1 = BatchNormalization()(c4_1)
   c4_1 = actv()(c4_1)
   c4_2 = Conv2D(depth//4, (1, kernel_size), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_1)
   c4_2 = BatchNormalization()(c4_2)
   c4_2 = actv()(c4_2)
   c4_3 = Conv2D(depth//4, (kernel_size, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_2)
   c4_3 = BatchNormalization()(c4_3)
   c4_3 = actv()(c4_3)
   c4_4 = Conv2D(depth//4, (1, kernel_size), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_3)
   c4_4 = BatchNormalization()(c4_4)
   c4_4 = actv()(c4_4)
   c4_5 = Conv2D(depth//4, (kernel_size, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_4)
   c4_5 = BatchNormalization()(c4_5)
   c4_5 = actv()(c4_5)
   
   res = concatenate([c1_1, c2_1, c3_3, c4_5])
   res = BatchNormalization()(res)
   res = actv()(res)
   return res

# used only in coarsest 6X6 grid to promote higher dimensional representations 
def expanded_filter_bank(inputs,depth,batch_mode=0, splitted=False, activation='relu'):
   actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None 
   
   c1_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
   
   pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', data_format='channels_last')(inputs)
   c2_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(pool)
   
   c3_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
   c3_1 = BatchNormalization()(c3_1)
   c3_1 = actv()(c3_1)
   c3_2 = Conv2D(depth//4, (1, 3), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_1)
   c3_3 = Conv2D(depth//4, (3, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c3_1)
   
   c4_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
   c4_1 = BatchNormalization()(c4_1)
   c4_1 = actv()(c4_1)
   c4_2 = Conv2D(depth//4, (3, 3), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_1)
   c4_2 = BatchNormalization()(c4_2)
   c4_2 = actv()(c4_2)
   c4_3 = Conv2D(depth//4, (1, 3), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_2)
   c4_4 = Conv2D(depth//4, (3, 1), kernel_initializer='he_normal', padding='same', data_format='channels_last')(c4_2)   
   res = concatenate([c1_1, c2_1, c3_2, c3_3, c4_3, c4_4])
   res = BatchNormalization()(res)
   res = actv()(res)
   return res
 

def get_unet(lr_in=1e-5,):
    act = 'elu'
    kernel_size=(2*2)-(2%2)
    inputs = Input((img_rows, img_cols, 1))
    conv1 = inception_block(inputs,32,1,batch_mode=2, splitted=False, activation=act)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = inception_block(pool1,64,1,batch_mode=2, splitted=False, activation=act)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = inception_block(pool2,128,2,batch_mode=2, splitted=True, activation=act)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = inception_block(pool3,256,2,batch_mode=2, splitted=True, activation=act)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = assymetrical_inception(pool4,512,7,batch_mode=2, splitted=True, activation=act)
    # no pooling after conv5
    
    from_c4 = residual_block(conv4,256)
    up_from_c5 = Conv2DTranspose(filters=512,kernel_size=kernel_size,strides=(2, 2),
                                 kernel_initializer='he_normal',padding='same')(conv5)
    inp_c6 = concatenate([from_c4,up_from_c5],axis=-1)
    conv6 = inception_block(inp_c6,256,2,batch_mode=2, splitted=True, activation=act)
    
    from_c3 = residual_block(conv3,128)
    up_from_c6 = Conv2DTranspose(filters=256,kernel_size=kernel_size,strides=(2, 2),
                                 kernel_initializer='he_normal',padding='same')(conv6)
    inp_c7 = concatenate([from_c3,up_from_c6],axis=-1)
    conv7 = inception_block(inp_c7,128,2,batch_mode=2, splitted=True, activation=act)
    
    from_c2 = residual_block(conv2,64)
    up_from_c7 = Conv2DTranspose(filters=128,kernel_size=kernel_size,strides=(2, 2),
                                kernel_initializer='he_normal', padding='same')(conv7)
    inp_c8 = concatenate([from_c2,up_from_c7],axis=-1)
    conv8 = inception_block(inp_c8,64,1,batch_mode=2, splitted=False, activation=act)
    
    from_c1 = residual_block(conv1,32)
    up_from_c8 = Conv2DTranspose(filters=64,kernel_size=kernel_size,strides=(2, 2),
                                 kernel_initializer='he_normal',padding='same')(conv8)
    inp_c9 = concatenate([from_c1,up_from_c8],axis=-1)
    conv9 = inception_block(inp_c9,32,1,batch_mode=2, splitted=False, activation=act)
    
    conv10 = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same',
                    data_format='channels_last')(conv9)
    
    
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=lr_in), loss=dice_coef_loss, metrics=[dice_coef])

    return model
