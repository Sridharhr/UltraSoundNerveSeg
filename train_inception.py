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
   
def get_unet():
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

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# returns a boolean mask which is set true for all image indices which has a valid mask
def get_object_existence(mask_array):
    return np.array([np.sum(mask_array[i]) != 0 for i in range(mask_array.shape[0])])


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    existence_mask =  get_object_existence(imgs_mask_train)

    imgs_train = preprocess(imgs_train)
    

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std
    
    y_train = existence_mask.astype(np.uint8) 
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss')
    model_save_best = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    #early_s = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    # TODO: experiment with different batch sizes
    # TODO: try k-fold ensemble training
    model.fit(imgs_train, y_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint, model_save_best])
    exit

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
