# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from skimage.transform import resize
import numpy as np


from data_ensemble import load_train_data, load_test_data
from inception_net_segment import get_unet

img_rows = 192
img_cols = 208
# returns a boolean mask which is set true for all image indices which has a valid mask
def get_object_existence(mask_array):
    return np.array([np.sum(mask_array[i]) != 0 for i in range(mask_array.shape[0])])


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

imgs_train, imgs_mask_train, patient_ids = load_train_data()
imgs_train = preprocess(imgs_train)
imgs_mask_train = preprocess(imgs_mask_train)
imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train/= 255.0     # done becaosue final layer is sigmoid
existence_mask =  get_object_existence(imgs_mask_train)

imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

y_train = existence_mask.astype(np.uint8) 

print('-'*30)
print('Creating and compiling model...')
print('-'*30)
#model_save_best = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)


# create model
model = KerasClassifier(build_fn=get_unet, epochs=20, batch_size=32, verbose=1)
epochs=[5,10,15,20]
batches = [8, 16, 32, 64]
lr = [1e-3,1e-4,1e-5,1e-6]
param_grid = dict(epochs=epochs, batch_size=batches, lr_in=lr)
#TODO: try out setting iid = False below
grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result = grid.fit(imgs_train, imgs_mask_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
