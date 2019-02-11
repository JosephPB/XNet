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

image_size = 200
train_size = 0.7
n_classes = 3

hdf5_path = "final"

# output hdf5 file
hdf5_name = '_'.join(data_to_add) 

hdf5_name = hdf5_name + '_s' + str(image_size) + '.hdf5'


# ******************* TRAIN/TEST/VAL **********************#

# Get balanced body parts split into train test and validation sets
images_train, labels_train, body_train, filenames_train, images_test, labels_test, body_test, \
filenames_test, images_val, labels_val, body_val, filenames_val = \
balanced_test_val_split(main_path, data_to_add, image_size, train_size, n_classes)

# Save hdf5 file without augmentations
create_h5.write_h5(hdf5_name, images_train, labels_train/255, body_train,filenames_train, images_test, labels_test/255,body_test,filenames_test,\
		images_val, labels_val/255,body_val ,filenames_val)

