from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Convolution2D
from keras.layers import BatchNormalization, Reshape, Layer
from keras.layers import Activation, Flatten, Dense, ConvLSTM2D, LeakyReLU
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras import losses
from keras.models import load_model

def model(input_shape=(64,64,3), classes=3, kernel_size = 3, filter_depth = (64,128,256,512,0)):
    
    img_input = Input(shape=input_shape)
    x = img_input
    
    # Encoder
    x = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x1 = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x1)
    #100x100
    
    x = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x2 = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x2)
    #50x50
    
    x = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x3 = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x3)
    #25x25
    
    #Flat
    x = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #25x25
    
    x = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #25x25
    
    #Up
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    y3 = Activation("relu")(x)
    x = Concatenate()([x3,y3])
    #50x50
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    y2 = Activation("relu")(x)
    x = Concatenate()([x2,y2])
    #100x100
    
    #Down
    x = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    u2 = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(u2)
    #50x50
    
    x = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    u3 = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(u3)
    
    #25x25
    
    #Flat
    x = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #25x25
    
    x = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #25x25
    
    #Encoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    z3 = Activation("relu")(x)
    x = Concatenate()([u3,z3])
    #50x50
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    z2 =  Activation("relu")(x)
    x = Concatenate()([u2,z2])
    #100x100
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(x)
    x = BatchNormalization()(x)
    z1 = Activation("relu")(x)
    x = Concatenate()([x1,z1])
    #200x200
    
    x = Conv2D(classes, (1,1), padding="valid")(x)
    
    
    x = Reshape((input_shape[0]*input_shape[1],classes))(x)
    x = Activation("softmax")(x)
    
    model = Model(img_input, x)

    return model