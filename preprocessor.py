import os
from PIL import Image
import numpy as np
import os.path as path
import cv2
from os.path import expanduser

'''
    author/s: Keiron O'Shea, mona
    description: preproccesses the images for standard height and width
'''
import cv2

def load_directories(directory):
	image_list = []
	for dir_name,dir_names,file_names in os.walk(directory):
		for file in file_names:
			label = os.path.basename(os.path.normpath(dir_name))
			image_list.append([label, file])
	return image_list

def load_images(directory):
    image_list = []
    for dir_name, dir_name, file_names in os.walk(directory):
        for file in file_names:
            image_list.append([file])
    return image_list



def align_to_square(image, custom_height, custom_width):
    width,height = image.size
    if height < width:
        filler = (width - height)/2
        image_array = np.asarray(image)
        new_image_array = np.zeros((custom_height, custom_width, 3), dtype="uint8")
        new_image_array[filler:filler+height,:,:]=image_array[:,:,:]
        image = Image.fromarray(new_image_array, "RGB")
    if height > width:
        length = height - width
        image_array = np.asarray(image)
        image = Image.fromarray(image_array[0:height-length,:,:],"RGB")
    return image


def preprocess(directory, custom_directory, custom_height, custom_width):
    
    save_path = custom_directory
    print save_path
    print directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_list = load_directories(directory)
    #print directory+"/"+image_name[0]+"/"+image_name[1]
    #print image_list
    for image_name in image_list:
        #print image_name[0]+'/'+image_name[1]
        image = Image.open(directory+"/"+image_name[0]+"/"+image_name[1])
        #print directory+"/"+image_name[0]+"/"+image_name[1]
        # image = crop_images(image) #Temporary "fix"
        width, height = image.size
        image = image.resize((custom_height, custom_width*height/width))
        image = align_to_square(image, custom_height, custom_width)
        if not os.path.exists(save_path + "/" + image_name[0]):
            os.makedirs(save_path + "/" + image_name[0])
        image.save(save_path+"/"+image_name[0]+"/"+image_name[1])

import pdb
if __name__ == "__main__":
    home_directory = expanduser("~")
    custom_height = 128
    custom_width = 128
    custom_dir = home_directory + "/keras-master/Plants/Preprocessed/FilteredCLEF"
    directory = home_directory + "/keras-master/Plants/FilteredCLEF"
    preprocess(directory, custom_dir, custom_height, custom_width);
  




