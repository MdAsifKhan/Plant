'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python myCNNTest.py
        THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu,floatX=float32 python myCNNTest.py
    CPU run command:
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from os.path import expanduser
from sklearn import metrics
from sklearn.metrics import classification_report
import os
import imp, numpy as np
import time
import imp, os.path as path

import pdb

pyvec_loader = imp.load_source('loader', 'loader.py')

def load_data(directory, custom_height, custom_width, split, withTest):
    print "Loading the data...\n"
    train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes = pyvec_api.load_images \
        (directory,custom_height,custom_width, split, withTest)
    return train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes
    train_data, train_label, val_data, val_label, test_data, test_label, num_classes = pyvec_api.load_images \
        (directory,custom_height,custom_width, split, withTest)
    return train_data, train_label, val_data, val_label, test_data, test_label, test_images_name, num_classes


def create_model(num_classes):
    print "Creating the model...\n"
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, border_mode='valid', input_shape=(3, 128, 128)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def get_data(directory):
    all_list = []
    test_list =  os.listdir(directory)
    for test in test_list:
        test_list1 = os.listdir(directory+test)
        for test1 in test_list1:
            all_list.append(directory+test+'/'+test1)
    return all_list


def run_experiments(model, train_data, train_label, test_data, test_label, batch_size, num_epoch, num_classes):
    print "Doing some training and validation..", "with: ", num_epoch, " epochs"
 
    model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=num_epoch,
            show_accuracy=True, verbose=1, shuffle=True, validation_data=(test_data, test_label))

    print "\nAnd now the test (with", len(test_label),"samples)..."
    score = model.evaluate(test_data, test_label, show_accuracy=True, verbose=1, batch_size=batch_size)
    print "Test Accuracy: ", score[1]
    Ytest = np.nonzero(test_label)
    Ytest = Ytest[1].tolist()
    YPredict = model.predict_classes(test_data, verbose=0)
    YPredict = YPredict.tolist()
    confusion_mat = metrics.confusion_matrix(Ytest, YPredict)
    print "confusion matrix: \n",confusion_mat

    print "Classification report:\n"

    print(classification_report(Ytest, YPredict))
    return model

    # run tests on herb 
def test_model(model, data, label):
    YPredict = model.predict_classes(data, verbose=0)
    true_label = np.nonzero(label)
    true_label = true_label[1]

    return YPredict, true_label


if __name__ == "__main__":
    custom_height = 128
    custom_width = 128
    DATA_ROOT= 'Preprocessed/FilteredCLEF/'
    data_list = get_data(DATA_ROOT)

    np.random.seed(1337) # Reproducible results :)
    num_epoch = 40
    batch_size = 60
    traits = []
    models_dir = "FilteredCLEFModel/"
    test_results = "AllViewsResults4/"


    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(test_results):
        os.makedirs(test_results)
    split = 0.20
    train_data, train_label, test_data, test_label, num_classes,train_images_name,test_images_name = pyvec_loader.load_images(DATA_ROOT, custom_height, custom_width, split)
    
    model = create_model(num_classes)
    learned_model = run_experiments(model, train_data, train_label, test_data, test_label, batch_size, num_epoch, num_classes)

    model_name = models_dir + 'bestmodel.hdf5'
    learned_model.save_weights(model_name, overwrite=True)

    #performance on test 
    YPredict, true_label = test_model(learned_model, test_data, test_label)
    file_name = 'test_results.txt'
    with open(file_name,"w") as file1:
        for i,k in enumerate(test_images_name):
            file1.write(k+': '+str(YPredict[i])+', '+str(true_label[i])+'\n')


