from __future__ import print_function

import os
import numpy as np
import cv2

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
    patient_ids = []
    mask_stat = {}
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        print(image_name)
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        kernel = np.ones((3,3),np.uint8)
        img = cv2.fastNlMeansDenoising(img,10,10,7,21)
        # may be try a ksize=3 also
        img = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
        img = cv2.dilate(img,kernel,iterations = 1)
        cv2.imwrite('C:\\Users\\IBM_ADMIN\\Documents\\ML\\Kaggle\\UltraSound\\transform\\'+
                    image_name.split('.')[0] + '.jpg',img)
        #img = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

        img = np.array([img])
        img_mask = np.array([img_mask])
        

        imgs[i] = img
        imgs_mask[i] = img_mask
        image_name_map.append(image_name)
        patient_ids.append(image_name.split('_')[0])
        pat_id = image_name.split('_')[0]
        if image_name.split('_')[0] not in mask_stat.keys():
            mask_stat[pat_id] = [0,0]
        if np.sum(img_mask) == 0:
            mask_stat[pat_id][1] = mask_stat[pat_id][1] + 1
        else:
            mask_stat[pat_id][0] = mask_stat[pat_id][0] + 1
            
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    print("mask statistics ===============")
    for key in mask_stat.keys():
        print(key,":",mask_stat[key])
        

    patient_ids = np.array(patient_ids)        
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    np.save('patient_ids.npy',patient_ids)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    patient_ids = np.load('patient_ids.npy')
    return imgs_train, imgs_mask_train, patient_ids


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
    #create_test_data()
