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
from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 200
img_cols = 200

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv5)
    
    conv6 = expanded_filter_bank(pool5,1024, batch_mode=2, splitted=False, activation=act)

    flattened_layer = Flatten()(conv6)
    # TODO: add learning rate decay
    
    # smooth transitioning to avoid representational bottleneck
    fc_layer1 = Dense(512,activation='relu',kernel_initializer='he_normal',bias_initializer = Constant(0.1))(flattened_layer)
    fc_layer2 = Dense(256,activation='relu',kernel_initializer='he_normal',bias_initializer = Constant(0.1))(fc_layer1)
    fc_layer3 = Dense(128,activation='relu',kernel_initializer='he_normal',bias_initializer = Constant(0.1))(fc_layer2)
    fc_layer4 = Dense(64,activation='relu',kernel_initializer='he_normal',bias_initializer = Constant(0.1))(fc_layer3)
    fc_layer5 = Dense(32,activation='relu',kernel_initializer='he_normal',bias_initializer = Constant(0.1))(fc_layer4)
    fc_layer6 = Dense(16,activation='relu',kernel_initializer='he_normal',bias_initializer = Constant(0.1))(fc_layer5)
    out_layer = Dense(1,activation='sigmoid',kernel_initializer='he_normal')(fc_layer6)

    model = Model(inputs=[inputs], outputs=[out_layer])

    model.compile(optimizer=Adam(lr=lr_in), loss='binary_crossentropy', metrics=['accuracy'])

    return model
