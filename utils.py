import numpy as np
import random
import rasterio
import glob
import cv2
from random import shuffle
import os
import scipy.misc
import rgb2label as gen_label


# Utilities
def balanced_test_val_split(main_path, data_to_add, image_size, train_size, n_classes):
	images_found = []
	labels_found = []
	for category in data_to_add:
		
		print('Checking labels and data match in %s folder ...'%category)
		data_path =os.path.join( main_path , 'Images' , category )
		data_path += os.sep + '*.tif'
		 
		labels_path = os.path.join(main_path, 'Labels', category)
		labels_path += os.sep + '*.jpg'

		images = glob.glob(data_path)
		labels =  glob.glob(labels_path)
		assert len(labels) != 0
		#print('Checking if number of labeled files matches number of data image files....')
		# Check that number of labels corresponds to number of images

		assert len(labels) == len(images)

		# Check that they have the same names
		label_filename = []
		img_filename = []

		for (i, img) in enumerate(images):
			label_filename.append(labels[i].split(os.sep)[-1].split('.')[0].replace('onehot', ''))
			img_filename.append(img.split(os.sep)[-1].split('.')[0]  )


		label_filename = sorted(label_filename)
		img_filename = sorted(img_filename)
		
		for i in range(len(label_filename)):

			assert label_filename[i] == img_filename[i]
			images_found.append(  os.path.join(main_path , 'Images' , category) + os.sep + img_filename[i] + '.tif')
			labels_found.append(  os.path.join(main_path , 'Labels' , category) + os.sep + label_filename[i] + '.jpg')


		print('Names of labels and data in folder %s match perfectly, %d images found . '%(category, len(img_filename)))

	#shuffle images and labels 
	c = list(zip(images_found,labels_found))
	shuffle(c)
	images, labels = zip(*c)

	# Read and save all images + labels + bodypart
	images_read = np.zeros((len(images),image_size,image_size,1),dtype=np.float32)
	labels_read = np.zeros((len(labels), image_size, image_size,3),dtype=np.uint8)
	bodyparts = np.empty((len(images)),'S10')
	split_names = np.empty((len(images)),'S50')
	for i in range(len(images)):
		filename = images[i]
		img = rasterio.open(filename)
		img = img.read(1)
		images_read[i,...,0] = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

		label_filename = labels[i]
		labels_images = cv2.imread(label_filename)

		labels_read[i,...] = scipy.misc.imresize(labels_images, (image_size,image_size,3), interp='nearest', mode=None)
		labels_read[i,...] = np.uint8(labels_read[i,...])
		labels_read[i,...] = 255*gen_label.get_categorical_label(labels_read[i,...], n_classes)

		# Clean bodyparts names
		bodypart = filename.split(os.sep)[-1].split('_')[0].lower()
		split_names[i] = filename.split(os.sep)[-1].split('.')[0].lower()
		if((bodypart == 'left') or (bodypart == 'right') or (bodypart == 'asg')):
			bodypart = filename.split(os.sep)[-1].split('_')[1]
			if(bodypart == 'fractured'):
				bodypart = filename.split(os.sep)[-1].split('_')[2]
			if(bodypart == 'lower'):
				bodypart = filename.split(os.sep)[-1].split('_')[2]
		if((bodypart == 'belly') or (bodypart == 'plate')):
				bodypart = filename.split(os.sep)[-1].split('_')[1]
		if((bodypart == 'leg') and (filename.split(os.sep)[-1].split('_')[1] == 'lamb')):
				bodypart = filename.split(os.sep)[-1].split('_')[1]              
		# Remove numbers
		bodypart = ''.join(i for i in bodypart if not i.isdigit())
		if(bodypart == 'nof'):
			bodypart = 'neckoffemur'
		bodypart = bodypart.split('.')[0]
		if(bodypart == 'anke'):
			bodypart = 'ankle'
			
		if(bodypart == 'lumbar'):
			bodypart = 'lumbarspin'
		bodypart = bodypart.encode("ascii", "ignore")
		bodyparts[i] = bodypart


	unique, counts = np.unique(bodyparts, return_counts=True)
	unique_per_category = dict(zip(unique, counts))

	#print('There are %d different bodyparts'%len(unique_per_category))

	indices = np.arange(images_read.shape[0])


	# Build balanced test and validation sets
	one_per_class = []
	for i in unique_per_category:
		split_category = np.where(bodyparts==i)[0].tolist()
		#pick one from each category to be part of the test set
		chosen_one_per_class = random.choice(split_category)
		indices_to_remove = np.argwhere( indices ==chosen_one_per_class)[0].tolist()
		indices = np.delete(indices, indices_to_remove)
		one_per_class.append(chosen_one_per_class)

	bodyparts_cut = bodyparts[indices]
	unique, counts = np.unique(bodyparts_cut, return_counts=True)
	unique_per_category = dict(zip(unique, counts))

	#print('Test that they are unique')
	#print(len(one_per_class) == len(set(one_per_class)))
	# From the different bodyparts left fill the test set from those that have more than one example
	# until test size is 0.3*total

	extra_need = int((1-train_size)*len(images)) - len(one_per_class)

	counter = 0
	test_extra = []
	while ( counter < extra_need ):
		#reshuffle dictionary
		keys = list(unique_per_category.keys())
		np.random.shuffle(keys)
		for bodypart in keys:
			if ( counter >= extra_need):
				break
			if( unique_per_category[bodypart] == 1 or unique_per_category[bodypart] == 0):
				continue

			#get random sample of that bodypart
			bodypart_indices = np.where(bodyparts[indices] == bodypart)[0].tolist()
			bodypart_choice = random.choice(indices[bodypart_indices])
			test_extra.append(bodypart_choice)
			#remove bodypart index to avoid repetition
			unique_per_category[bodypart] -= 1
			remove_bodypart_index = np.argwhere( indices == bodypart_choice)[0].tolist()
			indices = np.delete(indices, remove_bodypart_index )
			counter += 1
		 
	test_indices = np.concatenate((one_per_class,test_extra))

	images_train = images_read[indices,...]
	body_train = bodyparts[indices]
	split_names_train = split_names[indices]
	labels_train = labels_read[indices,...]

	random.shuffle(test_indices)

	images_test = images_read[test_indices[:int(len(test_indices)/2)],...]
	body_test = bodyparts[test_indices[:int(len(test_indices)/2)]]
	split_names_test = split_names[test_indices[:int(len(test_indices)/2)]]
	labels_test = labels_read[test_indices[:int(len(test_indices)/2)],...]

	images_val = images_read[test_indices[int(len(test_indices)/2):],...]
	body_val = bodyparts[test_indices[int(len(test_indices)/2):]]
	split_names_val = split_names[test_indices[int(len(test_indices)/2):]]
	labels_val = labels_read[test_indices[int(len(test_indices)/2):],...]


	#print(np.in1d(split_names_test, split_names_val, assume_unique=False, invert=False))
	#print('FINAL SHAPES')
	#print('train set :  %d images'%images_train.shape[0])
	#print('test set :  %d images'%images_test.shape[0])
	#print('val set :  %d images'%images_val.shape[0])

	# Check that we didn't lose images on the way
	assert (images_train.shape[0] + images_test.shape[0] + images_val.shape[0]) == len(images)

	return images_train, labels_train, body_train, split_names_train, images_test, labels_test, body_test,\
	split_names_test, images_val, labels_val, body_val, split_names_val 




def shuffle_together_simple(images, labels, bodyparts):

    c = list(zip(images,labels, bodyparts))
    shuffle(c)
    images, labels, bodyparts = zip(*c)    
    images = np.asarray(images)
    labels = np.asarray(labels)
    bodyparts = np.asarray(bodyparts)
    
    return images, labels, bodyparts

def shuffle_together(images, labels, bodyparts, filenames):

    c = list(zip(images,labels, bodyparts,filenames))
    shuffle(c)
    images, labels, bodyparts, filenames = zip(*c)    
    images = np.asarray(images)
    labels = np.asarray(labels)
    bodyparts = np.asarray(bodyparts)
    filenames = np.asarray(filenames)
    
    return images, labels, bodyparts, filenames


def random_crop(x, y, permin, permax):
    h, w, _ = x.shape
    per_h = random.uniform(permin, permax)
    per_w = random.uniform(permin, permax)
    crop_size = (int((1-per_h)*h),int((1-per_w)*w))

    rangew = (w - crop_size[0]) // 2 if w>crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h>crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1], :]
    cropped_y = y[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1], :]
    resize_x = cv2.resize(cropped_x, (h, w), interpolation=cv2.INTER_CUBIC)
    resize_y = cv2.resize(cropped_y, (h, w), interpolation=cv2.INTER_NEAREST)
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return np.reshape(resize_x,(h,w,1)), resize_y

