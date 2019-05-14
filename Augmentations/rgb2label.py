import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
import glob
from keras.utils import to_categorical

    

def get_mask_from_color( image, color ):
    """ Given one image and one color, returns a mask of the same shape as the image, with True values on the pixel positions with the same specified color"""
    rows, columns, channels = image.shape
    total_pixels = rows * columns
    image_flat = image.reshape(total_pixels, channels)
    color_array = np.array([color,] * total_pixels)
    channels_mask = np.isclose(image_flat, color_array, atol = 100)
    #combine channels
    mask = np.logical_and(channels_mask[:,0], channels_mask[:,1])
    mask = np.logical_and(mask, channels_mask[:,2])
    return mask.reshape(rows,columns)

def get_012_label(image, n_colors = 3, colors = [[255,255,255], [255,255,0], [0,0,255]]):
    """ Given one image, returns labeling 0,1,2 for 3 colours."""
    #color_0 = [255,255,255]
    #color_1 = [255,255,0]
    #color_2 = [0,0,255]
    
    label_012 = np.zeros((image.shape[0], image.shape[1]))
    
    if(n_colors == 2):
        mask = get_mask_from_color(image, colors[2])
        label_012[mask] = 1
        
    elif(n_colors == 3):
        mask = get_mask_from_color(image, colors[1])
        label_012[mask] = 1
        mask = get_mask_from_color(image, colors[2])
        label_012[mask] = 2
    
    else:
        print("number of colors not implemented")
        return False

    return label_012

def get_categorical_label(image, n_classes = 3):
    """ Given an image, computes the 012 label and uses keras to compute the categorical label"""
    label_012 = get_012_label(image, n_classes)
    return to_categorical(label_012, n_classes)
