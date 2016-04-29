import os, random
import numpy as np
from PIL import Image
from keras.utils import np_utils
from collections import Counter
import os, cv2
from sklearn.cross_validation import StratifiedShuffleSplit


def get_class_size(directory):
    nb_classes = len([classes for classes in os.listdir(directory) if os.path.isdir(directory)])
    return nb_classes


def rotate_images(directory, image_list, image_height, image_width, augment=False):

    if augment == True:	
	    rotation_matrix_1 = cv2.getRotationMatrix2D((image_height/2,image_width/2), 4, 1)
	    rotation_matrix_2 = cv2.getRotationMatrix2D((image_height/2,image_width/2), 7, 1)
	    rotation_matrix_3 = cv2.getRotationMatrix2D((image_height/2,image_width/2), -7, 1)
	    rotation_matrix_4 = cv2.getRotationMatrix2D((image_height/2,image_width/2), -4, 1)
	    augmented_image_list = []
	    for images in image_list:
		 image = cv2.imread(directory+"/"+images[0]+"/"+images[1])
		 augmented_image_list.append(images[0]+"/"+images[1])
            return augmented_image_list 
    if augment== False: 
	    augmented_image_list = []
	    for images in image_list:
		 image = cv2.imread(directory+"/"+images[0]+"/"+images[1])
		 augmented_image_list.append(images[0]+"/"+images[1])
	    return augmented_image_list


def get_labels(directory):
    imglist = []
    for dirname,dirnames,filenames in os.walk(directory):
        for filename in filenames:
            label = os.path.basename(os.path.normpath(dirname))
            imglist.append([label, filename])
    class_sizes = get_class_size(directory)
    return imglist, class_sizes

def split_images_dataset(split,images_list, with_test):
    if with_test == True:
        #print len(images_list)
        number_train_images = len(images_list) * split
        #print number_train_images
        number_test_and_validation = len(images_list) * ((1-split)/2)
        #print number_test_and_validation 
        train_list = images_list[0:int(number_train_images)]
        #valid_list = images_list[int(number_train_images):int(number_train_images+number_test_and_validation)]
        test_list = images_list[int(number_train_images):]
        return train_list,test_list
    else:
        number_train_images = len(images_list) * split
        number_test_and_validation = len(images_list) * (1-split)
        #print number_test_and_validation 
        train_list = images_list[0:int(number_train_images)]
        valid_list = images_list[int(number_train_images):]
        return train_list,valid_list
import pdb
def load_images(directory, image_height, image_width, split=0.15):
    # Retrieves a list of images.
    image_list,nb_classes = get_labels(directory)
    random.shuffle(image_list)
    print "num of classes:", nb_classes
    #augment images by rotation 
    data_list = rotate_images(directory, image_list, image_height, image_width, False)

    data, labels, names = vectorize(directory, data_list, image_height, image_width, nb_classes)
    train_data, train_label, test_data, test_label, train_names, test_names = split_stratified(data, labels, split, names)
    train_label = np_utils.to_categorical(train_label, nb_classes)
    test_label = np_utils.to_categorical(test_label, nb_classes)
	
    return train_data, train_label, test_data, test_label, nb_classes, train_names, test_names


def vectorize(directory, image_list, image_height, image_width, nb_classes):
    data = np.empty((len(image_list), 3, image_height, image_width), dtype="float32")
    data.flatten()
    # Creates an array, ready for the labels to go into vectors to correlate with training_data
    label = np.empty((len(image_list),), dtype="uint8")
    images_names = np.empty((len(image_list),), dtype= object)
    
    dict = {'Branch':0,'Entire':1,'Flower':2,'Fruit':3,'Leaf':4,'LeafScan':5,'Stem':6}

    for i, image_name in enumerate(image_list):
        # Open the files.
        images = Image.open(directory+"/"+image_name)
        images = images.resize((image_height, image_width), Image.ANTIALIAS)
        # Converts the images into float32 representation
        vectored_image = np.asarray(images, dtype="float32")
        # 3 shape vector..
        data[i,:,:,:] = [vectored_image[:,:,0],vectored_image[:,:,1],vectored_image[:,:,2]]
        string = image_name[:image_name.index('/')]
        #Assigns label to the image.
        label[i] = int(dict[string])
        images_names[i] = image_name
    label = np_utils.to_categorical(label, nb_classes)
    data = data.reshape(data.shape[0], 3, image_height, image_width)/255
    data = data.astype("float32")
    return data, label, images_names


def split_stratified(data, label, split, image_names):

    #print "inside stratified!!"
    #print len(data)
    label = np.nonzero(label)
    label = label[1]

    sss = StratifiedShuffleSplit(label, 1, test_size=split, random_state=0)
    
    for train_index, test_index in sss:
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
 	train_names, test_names = image_names[train_index], image_names[test_index]

    #print "after loop"
    #print len(train_data)	
    return train_data, train_label, test_data, test_label, train_names, test_names

