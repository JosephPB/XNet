import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import glob
import pandas as pd
#import PIL
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import  * 
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras import losses
from keras.models import load_model as keras_load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import sys
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_curve, auc


sys.path.insert(0, '../')



class PostProcessing:
    beam =  0
    tissue = 1
    bone = 2
    def __init__(self, model_path, dataset_path, loss = 'categorical_crossentropy', device = "cpu"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.read_h5_file()
        print(loss)
        self.load_model(device = device, loss = loss)
        print('Model loaded.')
        self.prediction_prob_rs, self.prediction_argmax = self.predict(device=device)

    def read_h5_file(self):
        "Read data from h5file"
        dataset = h5py.File(self.dataset_path, 'r')
        self.train_images = dataset['train_img']
        self.test_images =  dataset['test_img'][:]
        self.val_images =  dataset['val_img'][:]
        self.train_labels = dataset['train_label']
        self.train_body = dataset['train_bodypart'][:]
        self.test_labels = dataset['test_label'][:]
        self.val_labels = dataset['val_label'][:]
        self.test_body = dataset['test_bodypart'][:]
        self.val_body = dataset['val_bodypart'][:]
        self.test_filenames = dataset['test_file'][:]
        self.val_filenames = dataset['val_file'][:]
        self.no_images_training, self.height, self.width, self.classes = self.train_labels.shape
        self.train_labels = np.reshape(self.train_labels, (-1,self.height*self.width ,self.classes))
        self.test_labels = np.reshape(self.test_labels, (-1,self.height*self.width ,self.classes))
        self.val_labels = np.reshape(self.val_labels, (-1,self.height*self.width ,self.classes))
        self.test_images = np.concatenate((self.test_images, self.val_images))
        self.test_labels = np.concatenate((self.test_labels, self.val_labels))
        self.test_filenames = np.concatenate((self.test_filenames, self.val_filenames))
        #REMOVE breast and Rectangles
        #mask1 = np.where((self.test_filenames != b'breast_phantom') & (self.test_filenames != b'pmmaandal'))
        #self.test_images = self.test_images[mask1]
        #self.test_labels = self.test_labels[mask1]
        
        dataset.close()
        
    def load_model(self, device = "cpu", optimizer = Adam(lr=1e-4), loss = "categorical_crossentropy",\
                  metrics = ['accuracy'] ):
        if(device == "cpu"):
            with tf.device("/cpu:0"):
                if(loss  == "jaccard"):
                    from jaccard_loss import jaccard_distance
                    self.model = keras_load_model(self.model_path,custom_objects ={'jaccard_distance': jaccard_distance})
                    self.model.compile(optimizer, loss = jaccard_distance, metrics = metrics)
                elif(loss == "fancy"):
                    from kerasfancyloss import fancy_loss
                    self.model = keras_load_model(self.model_path,custom_objects ={'fancy_loss': fancy_loss})
                    self.model.compile(optimizer, loss =fancy_loss, metrics = metrics)
                else:
                    self.model = keras_load_model(self.model_path)
                    self.model.compile(optimizer, loss, metrics)
        elif(device == "gpu"):
            if(loss  == "jaccard"):
                from jaccard_loss import jaccard_distance
                self.model = keras_load_model(self.model_path,custom_objects ={'jaccard_distance': jaccard_distance})
                self.model.compile(optimizer, loss = jaccard_distance, metrics = metrics)
            elif(loss == "fancy"):
                from kerasfancyloss import fancy_loss
                self.model = keras_load_model(self.model_path,custom_objects ={'fancy_loss': fancy_loss})
                self.model.compile(optimizer, loss =fancy_loss, metrics = metrics)
            else:
                self.model = keras_load_model(self.model_path)
                self.model.compile(optimizer, loss, metrics)
        else:
            print("Device not understood")
            return None
    
    def predict(self, device = "cpu", images = None):
        if(images is None):
            images = self.test_images
        if( device == "cpu"):
            with tf.device("/cpu:0"):
                prediction_prob = self.model.predict(images, batch_size=1)
        elif(device == "gpu"):
            prediction_prob = self.model.predict(images, batch_size=1)
        else:
            print("Device not found")
            return None
        prediction_prob_rs = prediction_prob.reshape((-1,self.classes))
        prediction_argmax = np.argmax(prediction_prob_rs, axis = -1)
        return prediction_prob_rs, prediction_argmax
    
    def evaluate_overall(self, device = "gpu"):
        images = self.test_images
        labels = self.test_labels
        if(device == "cpu"):
            with tf.device("/cpu:0"):
                loss_test, accuracy_test = self.model.evaluate(images,labels, batch_size = 1)
                
        elif(device == "gpu"):
            loss_test, accuracy_test = self.model.evaluate(images,labels, batch_size = 1)
        else:
            print("Device not understood")
            return None
       
        print("Overall accuracy : \n")
        print ('On test set {}%'.format(round(accuracy_test,2)*100))
        
        # Count number of trainable parameters
        trainable_count =  int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        print('Trainable params: {:,}'.format(trainable_count))
        return accuracy_test, trainable_count

    def evaluate_perclass(self, device = "gpu"):
        
        _, predictions = self.predict()
        labels = self.test_labels
        labels = np.argmax(labels, axis = -1)
        labels = labels.flatten()
        
        beam_gt = np.where(labels == self.beam)[0]
        beam_pred = np.where(predictions == self.beam)[0]
        beam_accuracy = float(len(np.intersect1d(beam_gt, beam_pred, assume_unique=True)))/float(len(beam_pred))
        
        tissue_gt = np.where(labels == self.tissue)[0]
        tissue_pred = np.where(predictions == self.tissue)[0]
        tissue_accuracy = float(len(np.intersect1d(tissue_gt, tissue_pred, assume_unique=True)))/float(len(tissue_pred))
    
        bone_gt = np.where(labels == self.bone)[0]
        bone_pred = np.where(predictions == self.bone)[0]
        bone_accuracy = float(len(np.intersect1d(bone_gt, bone_pred, assume_unique=True)))/float(len(bone_pred))
    
        print('Accuracy on the different classes : \n')
        print('Open beam %f, Soft tissue %f, Bone %f'%(beam_accuracy,tissue_accuracy, bone_accuracy))
        return beam_accuracy, tissue_accuracy, bone_accuracy
    
    def tpfp(self, predictions = None, single_index = -1):
        
        if (not (single_index == -1)):
            labels = self.test_labels[single_index]
            if(predictions is not None):
                prediction_argmax = predictions.reshape(-1,200,200)
            else:
                prediction_argmax = self.prediction_argmax.reshape(-1,200,200)
                
            prediction_argmax = prediction_argmax[single_index]
            prediction_argmax = prediction_argmax.flatten()
        else:
            if( predictions is not None):
                prediction_argmax = predictions
            else:
                prediction_argmax = self.prediction_argmax
            labels = self.test_labels
            
        labels = np.argmax(labels, axis = -1)
        labels = labels.flatten()
        
        beam_gt = np.where(labels == self.beam)[0]
        beam_pred = np.where(prediction_argmax == self.beam)[0]
            
        tissue_gt = np.where(labels == self.tissue)[0]
        tissue_pred = np.where(prediction_argmax == self.tissue)[0]

        if (len(tissue_pred) == 0):
            return 0,0


        bone_gt = np.where(labels == self.bone)[0]
        bone_pred = np.where(prediction_argmax == self.bone)[0]
        
        # FALSE POSITIVES
        false_positives = 0
        beam_as_tissue = float(len(np.intersect1d(beam_gt, tissue_pred, assume_unique=True)))/float(len(tissue_pred))
        false_positives = beam_as_tissue 
        bone_as_tissue = float(len(np.intersect1d(bone_gt, tissue_pred, assume_unique=True)))/float(len(tissue_pred))
        false_positives += bone_as_tissue
        
        # TRUE POSITIVES
        
        true_positives = 0
        true_positives = len(np.intersect1d(tissue_gt, tissue_pred, assume_unique=True))/len(tissue_gt)
        
        # FALSE NEGATIVES
        
        false_negatives = 0
        tissue_as_beam = float(len(np.intersect1d(tissue_gt, beam_pred, assume_unique=True)))/float(len(tissue_gt))
        false_negatives = tissue_as_beam
        tissue_as_bone = float(len(np.intersect1d(tissue_gt, bone_pred, assume_unique=True)))/float(len(tissue_gt))
        false_negatives += tissue_as_bone
            
    
        return true_positives, false_positives
    
        
    def thresholding(self,threshold, device = "cpu"):
        prob_prediction_tissue = self.prediction_prob_rs[...,self.tissue]
        tissue_pred = np.where((prob_prediction_tissue > threshold))[0]
        
        prediction_improved = self.prediction_argmax
        prediction_improved[tissue_pred] = self.tissue
        
        tissue_notsure = np.where((prob_prediction_tissue <= threshold))[0]
        openbeam_bone = self.prediction_prob_rs[...,[0,2]]
        prediction_improved[tissue_notsure] = 2 * np.argmax(openbeam_bone[tissue_notsure], axis = -1)
        self.prediction_argmax = prediction_improved
        return prediction_improved
   
    def thresholding_bodypart(self):
        
        unique, counts = np.unique(self.test_body, return_counts=True)
        thresholds = 0.6*np.ones(len(unique))
        thresholds_dict = dict(zip(unique, thresholds))
        thresholds_dict[b'ankle'] = 0.85
        thresholds_dict[b'hand'] = 0.99
        thresholds_dict[b'cropped'] = 0.99 
        thresholds_dict[b'foils'] = 0.99 
        thresholds_dict[b'lumbarspin'] = 0.99 
        thresholds_dict[b'neckoffemu'] = 0.9
        prediction_prob_rs, prediction_argmax = self.predict()
        prediction_argmax = prediction_argmax.reshape(-1, self.height, self.width)
        prediction_improved = np.zeros_like(prediction_argmax)
        
        test_images = self.test_images[...,0]
        for i,image in enumerate(test_images):
            bodypart = self.test_body[i]
            threshold = thresholds_dict[bodypart]
            prediction_prob = prediction_prob_rs[i]
            prob_prediction_tissue = prediction_prob[...,self.tissue]
            tissue_pred = np.where((prob_prediction_tissue > threshold))[0]
            
            prediction_improved[i] = prediction_argmax[i]
            prediction_improved[tissue_pred] = self.tissue
            tissue_notsure = np.where((prob_prediction_tissue <= threshold))[0]
            openbeam_bone = prediction_prob[...,[0,2]]
            prediction_improved[tissue_notsure] = 2 * np.argmax(openbeam_bone[tissue_notsure], axis = -1)
            
        return prediction_improved
            
            
    
    def pixel_dilation(self, dilation_factor, predictions = None, both = False):
        '''Dilates pixels if bone and/or soft tissue.
        Input:
            prediction: argmaxed images shape = (height,width)
            dilation_factor: number of pixels by which to dilate
            both: bool, if True dilates both open beam and bone, with preference for bone, if False dilates bone'''
        
        if( predictions is None):
            _, predictions = self.predict()
        predictions = predictions.reshape((-1, self.height, self.width))
        predictions = predictions.astype(np.float32)
        predictions_dilated = np.ones_like(predictions)
        
        prediction_bone = np.zeros_like(predictions)
        bone_indices = np.where(predictions == self.bone)
        prediction_bone[bone_indices] = self.bone
        prediction_bone = prediction_bone.reshape((-1, self.height, self.width))
        prediction_bone = prediction_bone.astype(np.float32)
        prediction_bone_dilated = np.zeros_like(predictions)
        
        for i,prediction in enumerate(prediction_bone):
            #remove small groups of bone
            kernel_opening = np.ones((10,10), np.uint8)
            bone_pred = np.where(prediction == self.bone)
            opening = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel_opening)
        
            #dilate image
            kernel_dilate = np.ones((dilation_factor, dilation_factor), np.uint8 )
            dilated = cv2.dilate(opening, kernel_dilate)
            prediction_bone_dilated[i,...] = dilated
        
        predictions_dilated[np.where(prediction_bone_dilated == self.bone)] = self.bone 
        predictions_dilated[np.where(predictions == self.beam)] = self.beam
        
        return predictions_dilated
        
        
        # batch, height, width = prediction.shape
        # cp = np.copy(prediction)
    
        #for k in range(batch):
        #    for i in range(dilation_factor,height-dilation_factor):
        #        for j in range(dilation_factor,width-dilation_factor):            
        #            if prediction[k,i,j] == bone:
        #                cp[k,i-dilation_factor:i+dilation_factor+1,j-dilation_factor:j+dilation_factor+1] = bone
        #            if both == True:
        #                if cp[k,i,j] == open_beam:
        #                    cp[k,i-dilation_factor:i+dilation_factor+1,j-dilation_factor:j+dilation_factor+1] = open_beam 
        #return cp
        
    def plot(self,threshold, dilation_factor):
        probability_map, prediction = self.predict()
        prediction_threshold = self.thresholding(0.9)
        prediction_dilation = self.pixel_dilation(dilation_factor, prediction_threshold)
        ntestimages = len(self.test_images)
        left  = 0.1  # the left side of the subplots of the figure
        right = 0.4    # the right side of the subplots of the figure
        bottom = 0.1   # the bottom of the subplots of the figure
        top = 0.9      # the top of the subplots of the figure
        wspace = 0.08   # the amount of width reserved for blank space between subplots
        hspace = 0.1   # the amount of height reserved for white space between subplots
        
        labels_plot = self.test_labels.reshape(-1, self.height, self.width, 3) * 255
        prediction = prediction.reshape(-1, self.height, self.width)
        prediction_threshold = prediction_threshold.reshape(-1, self.height, self.width)
        prediction_dilation = prediction_dilation.reshape(-1, self.height, self.width)
        
        for i, image in enumerate(self.test_images):
            print(self.test_filenames[i])
            fig=plt.figure(figsize=(50, 50), dpi= 80, edgecolor='k',frameon=False)
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
            index_show = i
            print(i)
            # Need TP/FP per image
            plt.subplot(ntestimages,5,1)
            plt.title('Image')
            plt.imshow(image[...,0],cmap='gray')
            plt.axis('off')
        
            plt.subplot(ntestimages,5,2)
            plt.title('Ground truth')
            plt.imshow(labels_plot[i])
            plt.axis('off')
        
            plt.subplot(ntestimages,5,3)
            plt.title('Prediction')
            plt.imshow(prediction[i])
            plt.axis('off')
        
            #plt.subplot(ntestimages,5,4)
            #plt.title('Probability map')
            #plt.imshow(probability_map[i])
            #plt.axis('off')
            
            plt.subplot(ntestimages,5,4)
            plt.title('Threshold')
            plt.imshow(prediction_threshold[i])
            plt.axis('off')
            
            plt.subplot(ntestimages,5,5)
            plt.title('Dilated')
            plt.imshow(prediction_dilation[i])
            plt.axis('off')
            plt.show()
    
    def learning_curve(self, path_to_csv):
	
        csv_file = pd.read_csv(path_to_csv)
        self.csv = csv_file
        epochs = self.csv['epoch']
        train_loss = self.csv['loss']
        val_loss = self.csv['val_loss']
        train_acc = self.csv['acc']
        val_acc = self.csv['val_acc']
		
        train_err = 1 - train_acc
        val_err = 1 - val_acc
        fig, ax = plt.subplots(2,1, figsize=(15,15))
        ax[0].plot(epochs, train_err, color = 'blue', label = 'training error')
        ax[0].plot(epochs, val_err, color = 'orange', label = 'validation error')
        ax[0].plot(epochs, np.linspace(0.02,0.02,len(epochs)), color = 'green', label = 'desired error')
        ax[0].set_xlabel('number of epochs')
        ax[0].set_ylabel('error')
        ax[0].set_title('Error')
        ax[0].legend()
		
        ax[1].plot(epochs, train_loss, label = "training loss")
        ax[1].plot(epochs, val_loss, label = "validation loss")
        ax[1].legend()
        ax[1].set_xlabel("Number of epochs")
        ax[1].set_ylabel("Loss")
        ax[1].set_title("Loss")
		
        return fig, ax
        #plt.title('Epoch learning curve for Double Linked Network')
        #plt.savefig('Linked_epoch_LC.png', dpi = 250)


    def ROC_curve(self):
        # 1 for only tissue
        fpr, tpr, thresholds = roc_curve(self.test_labels[..., 1].reshape(-1), self.prediction_prob_rs[...,1].reshape(-1))
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr,
                 label='Tissue ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc),
                 color='indianred', linestyle=':', linewidth=4)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Some extension of Receiver operating characteristic to multi-class')
        ax.legend(loc="lower right")
        return fig, ax
