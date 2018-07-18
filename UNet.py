from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Reshape
from keras.layers import Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def model(input_shape=(64,64,3), classes=3, kernel_size = 3, filter_depth = (64,128,256,512,1024)):
    
    img_input = Input(shape=input_shape)

    #Encoder
    conv1 = Conv2D(filter_depth[0], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(img_input)
    conv1 = Conv2D(filter_depth[0], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filter_depth[1], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(filter_depth[1], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filter_depth[2], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(filter_depth[2], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(filter_depth[3], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(filter_depth[3], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(filter_depth[4], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(drop4)
    conv5 = Conv2D(filter_depth[4], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    #Decoder
    up6 = UpSampling2D(size=(2, 2))(pool4)
    conv6 = Conv2D(filter_depth[3], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(drop5)
    conv6 = Conv2D(filter_depth[3], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(filter_depth[2], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(filter_depth[2], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv7)
    
    up8 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D(filter_depth[1], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(filter_depth[1], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv8)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    copy9 = Concatenate()
    conv9 = Conv2D(filter_depth[0], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(filter_depth[0], (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, (kernel_size,kernel_size), activation = 'relu', padding = 'same')(conv9)
    
    x = Conv2D(classes, (1,1), padding="valid")(conv9)
    
    x = Reshape((input_shape[0]*input_shape[1],classes))(x)
    x = Activation("softmax")(x)
    
    model = Model(img_input, x)
    
    return model
