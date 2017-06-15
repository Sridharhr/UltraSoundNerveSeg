from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from shutil import copy

data_path = 'raw/'

image_rows = 420
image_cols = 580

# returns a boolean mask which is set true for all image indices which has a valid mask
def get_object_existence(mask_array):
    return np.array([np.sum(mask_array[i]) != 0 for i in range(mask_array.shape[0])])


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    # map to store the name if the ith image in imgs
    image_name_map = []
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
        image_name_map.append(image_name)

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    # filter maks and maskless images
    existence_mask = get_object_existence(imgs_mask)
    image_name = np.array(image_name_map)
    masked_images = image_name[existence_mask]
    non_masked_images = image_name[np.invert(existence_mask)]
    masked_train_length = int(0.8*len(masked_images))
    non_masked_train_length = int(0.8*len(non_masked_images))
    print(masked_train_length)
    print(non_masked_train_length)
    print(len(masked_images))
    print(len(non_masked_images))
    
 
    masked_images_train = masked_images[:masked_train_length]
    non_masked_images_train = non_masked_images[:non_masked_train_length]
    
    masked_images_cv = masked_images[masked_train_length:]
    non_masked_images_cv = non_masked_images[non_masked_train_length:]
    
    for image in masked_images_train:
        print('mask train ',image)
        copy('raw/train/'+image,'data/train/masked/')

    for image in non_masked_images_train:
        print('non_mask train ',image)
        copy('raw/train/'+image,'data/train/non_masked/')

    for image in masked_images_cv:
        print('mask cv ',image)
        copy('raw/train/'+image,'data/validation/masked/')

    for image in non_masked_images_cv:
        print('non_mask cv ',image)
        copy('raw/train/'+image,'data/validation/non_masked/')
        
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
