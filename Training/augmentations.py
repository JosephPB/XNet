from __future__ import division
from sklearn.model_selection import train_test_split
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import create_h5
import cv2
import glob
from random import shuffle
import h5py
import argparse
from keras.utils import to_categorical
import random
from keras.preprocessing.image import ImageDataGenerator
from utils import random_crop
from utils import shuffle_together
from utils import balanced_test_val_split
import sys
import time

# ******************* PARAMETERS *************************#
main_path = "Data"
data_to_add = ['Humans','CT','Phantom'] 
hdf5_path = "final"

EXAMPLES_PER_CATEGORY = 500
image_size = 200
train_size = 0.7
n_classes = 3


# output hdf5 file
hdf5_name = '_'.join(data_to_add) 

if(EXAMPLES_PER_CATEGORY == 0):
    hdf5_name = hdf5_name + '_s' + str(image_size) + '.hdf5'

else:
    hdf5_name =  hdf5_name +'_s'+str(image_size)+'_a'+ str(EXAMPLES_PER_CATEGORY)+ '.hdf5'


# ******************* TRAIN/TEST/VAL **********************#

# Get balanced body parts split into train test and validation sets
images_train, labels_train, body_train, filenames_train, images_test, labels_test, body_test, \
filenames_test, images_val, labels_val, body_val, filenames_val = \
balanced_test_val_split(main_path, data_to_add, image_size, train_size, n_classes)

# ******************* AUGMENTATIONS **********************#

# Find number of augmentations per image in order to have a balanced training set
unique, counts = np.unique(body_train, return_counts=True)
unique_per_category = dict(zip(unique, counts))
augmentations_per_category = dict(unique_per_category)
for key in unique_per_category:
    augmentations_per_category[key] = int(EXAMPLES_PER_CATEGORY/unique_per_category[key])

#Augmentation templates
translate_max = 0.01
rotate_max = 15
shear_max = 2

affine_trasform = iaa.Affine( translate_percent={"x": (-translate_max, translate_max),
                                                 "y": (-translate_max, translate_max)}, # translate by +-
                              rotate=(-rotate_max, rotate_max), # rotate by -rotate_max to +rotate_max degrees
                              shear=(-shear_max, shear_max), # shear by -shear_max to +shear_max degrees
                              order=[1], # use nearest neighbour or bilinear interpolation (fast)
                              cval=125, # if mode is constant, use a cval between 0 and 255
                              mode="reflect",
                              #mode = "",
                              name="Affine",
                             )


spatial_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5), affine_trasform])

other_aug = iaa.SomeOf((1, None),
        [
            iaa.OneOf([
                iaa.GaussianBlur((0, 0.4)), # blur images with a sigma between 0 and 1.0
                iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25), # very few

            ]),

        ])



augmentator = [spatial_aug,other_aug]
total_images=sum(augmentations_per_category[k]*unique_per_category[k] + unique_per_category[k] for k in augmentations_per_category)
images_aug = np.zeros((total_images,images_train.shape[1],images_train.shape[2],images_train.shape[3]))
labels_aug = np.zeros((total_images,labels_train.shape[1],labels_train.shape[2],labels_train.shape[3]))
bodypart = np.empty((total_images),dtype = 'S10')
filenames_aug = np.empty((total_images),dtype = 'S60')

images_aug[:images_train.shape[0],...] = images_train
labels_aug[:images_train.shape[0],...] = labels_train/255
bodypart[:images_train.shape[0],...] = body_train
filenames_aug[:images_train.shape[0],...] = filenames_train

# Loop  over the different bodyparts
counter = images_train.shape[0]
counter_block = 0
for i, (k, v) in enumerate(augmentations_per_category.items()):
    # Indices of images with a given bodypart
    indices = np.array(np.where(body_train == k )[0])
    # Number of augmentation per image
    N = int(v)

    for j in indices:
        for l in range(N):
            clear_output(wait=True)
            # Freeze randomization to apply same to labels
            spatial_det = augmentator[0].to_deterministic() 
            other_det = augmentator[1]

            images_aug[counter,...] = spatial_det.augment_image(images_train[j])

            labels_aug[counter,...] = spatial_det.augment_image(labels_train[j])
            img_crop, label_crop = random_crop(images_aug[counter,...],labels_aug[counter,...],0.1,0.4)
            images_aug[counter,...] = other_det.augment_image(img_crop )               
            labels_aug[counter,...] = to_categorical(np.argmax(label_crop,axis=-1))

            bodypart[counter] = k 
            
			# Save names of the augmented images starting with aug_
            filenames_aug[counter] = b'aug_' + filenames_train[j]
            sys.stdout.write('(Category %s) processing image %i/%i, augmented image %i/%i'%(k,counter_block,
                                                                                     body_train.shape[0],
                                                                                     l+1, N))
            sys.stdout.flush()
            time.sleep(0.5)
            counter +=1
        counter_block +=1

images_aug, labels_aug, bodypart, filenames_aug = shuffle_together(images_aug, labels_aug, bodypart, filenames_aug)

images_test, labels_test, body_test, filenames_test = shuffle_together(images_test, labels_test, body_test, filenames_test)

images_val, labels_val, body_val, filenames_val = shuffle_together(images_val, labels_val, body_val, filenames_val)

print('Finished playing with cadavers ! ')

create_h5.write_h5(hdf5_path + hdf5_name, images_aug, labels_aug, bodypart,filenames_aug, images_test, labels_test/255,body_test,filenames_test,\
		images_val, labels_val/255,body_val ,filenames_val)


print('Saving the hdf5 at %s ...'%hdf5_name)
