from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import  merge, Convolution2D, UpSampling2D
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.initializers import Constant
from keras import layers
from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 192
img_cols = 208

smooth = 1.

def residual_block(inputs,depth, activation='relu'):
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    residual = Conv2D(depth,(1,1),kernel_initializer='he_normal', padding='same', data_format='channels_last')(inputs)
    residual = BatchNormalization()(residual)
    residual = actv()(residual)
    residual = Conv2D(depth,(1,1),kernel_initializer='he_normal', padding='same', data_format='channels_last')(residual)
    residual = BatchNormalization()(residual)
    return layers.add([inputs,residual])



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

def get_unet():
    act = 'elu'
    kernel_size=(2*2)-(2%2)
    inputs = Input((img_rows, img_cols, 1))
    conv1 = inception_block(inputs,32,1,batch_mode=2, splitted=False, activation=act)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = inception_block(pool1,64,1,batch_mode=2, splitted=False, activation=act)
    #conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = inception_block(pool2,128,2,batch_mode=2, splitted=True, activation=act)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    #conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    #conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = inception_block(pool3,256,2,batch_mode=2, splitted=True, activation=act)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    #conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    #conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = assymetrical_inception(pool4,512,7,batch_mode=2, splitted=True, activation=act)

# no pooling after conv5
    from_c4 = residual_block(conv4,256)
    up_from_c5 = Conv2DTranspose(filters=512,kernel_size=kernel_size,strides=(2, 2),
                                 kernel_initializer='he_normal',padding='same')(conv5)
    inp_c6 = concatenate([from_c4,up_from_c5],axis=-1)
    conv6 = inception_block(inp_c6,256,2,batch_mode=2, splitted=True, activation=act)
   
    #up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    #conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    #conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
