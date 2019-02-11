from __future__ import division
import os, sys
import numpy as np
import cv2
import glob
from random import shuffle
from IPython.display import clear_output
import h5py
from sklearn.model_selection import train_test_split


def write_h5(hdf5_name,images_train,labels_train,body_train, file_train, images_test,labels_test,body_test ,\
        file_test,images_val,labels_val,body_val,file_val):

    hdf5_file = h5py.File(hdf5_name, mode='w')
    # Attributes
    hdf5_file.attrs['image_size'] = images_train.shape[2] 
    hdf5_file.attrs['max_value'] = 1.
    hdf5_file.attrs['min_value'] = 0.
    print(body_train.shape)
    # Datasets
    hdf5_file.create_dataset("train_img", images_train.shape, np.float64)
    hdf5_file.create_dataset("train_label", labels_train.shape, np.int)
    hdf5_file.create_dataset("train_bodypart", body_train.shape, 'S10')
    hdf5_file.create_dataset("train_file", file_train.shape, 'S60')

    hdf5_file.create_dataset("test_img", images_test.shape, np.float64)
    hdf5_file.create_dataset("test_label", labels_test.shape, np.int)
    hdf5_file.create_dataset("test_bodypart", body_test.shape, 'S10')
    hdf5_file.create_dataset("test_file", file_test.shape, 'S60')

    hdf5_file.create_dataset("val_img", images_val.shape, np.float64)
    hdf5_file.create_dataset("val_label", labels_val.shape, np.int)
    hdf5_file.create_dataset("val_bodypart", body_val.shape, 'S10')
    hdf5_file.create_dataset("val_file", file_val.shape, 'S60')
 
    categories = ['train','test','val']
    images_split = [images_train, images_test, images_val]
    labels_split =  [labels_train, labels_test, labels_val]
    bodys_split = [body_train, body_test, body_val]
    names_split = [file_train, file_test, file_val]
    for j  in range(len(images_split)):
        for i in range(images_split[j].shape[0]):
            clear_output(wait=True)
            # Zero mean
            img = images_split[j][i,...] - np.mean(images_split[j][i,...])
            # Normalization -> perform after augmentation
            img = (img-np.min(img))/(np.max(img) - np.min(img)) 
 
            hdf5_file[categories[j] + '_img'][i, ...] = img
            # same for labels
            #labels_simple = label_generate.GenerateOutput(labels_split[j][i,...])
            #labels_onehot = onehot.OneHot(labels_simple)
            #hdf5_file[categories[j] + "_label"][i, ...] = labels_onehot    
            hdf5_file[categories[j] + "_label"][i, ...] = labels_split[j][i,...] 
            hdf5_file[categories[j] + "_bodypart"][i] = bodys_split[j][i]
            hdf5_file[categories[j] + "_file"][i] = names_split[j][i]
            #print('Saving image %i/%i in %s path' %(i+1,images_split[j].shape[0], categories[j]))

    hdf5_file.close() 
