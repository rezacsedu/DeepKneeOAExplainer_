from __future__ import print_function
import numpy as np

np.random.seed(2668)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD
from random import shuffle
import time
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import xml.dom.minidom as xmldom
import os
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import sherpa

def load():
  imgList=[]
  labelList=[]
  files = os.listdir('D:/most/processedXR/side/training/') #training image path
  shuffle(files)
  for file in files:
       if file.endswith(".xml"):continue
       fi_d = os.path.join('D:/most/processedXR/side/training/',file) #training image path
       img=Image.open(fi_d)
       img=img.crop((0,0,1520,2047))
       im=np.array(img.resize((1520,2047), Image.ANTIALIAS))
       imgList.append(im)
       filename=file.split('.')[0]
       dom = xmldom.parse("D:/most/processedXR/side/training/"+filename+".xml") #training image path, where also contains ROI labels
       rois=dom.documentElement
       xmin = int(rois.getElementsByTagName("xmin")[0].firstChild.data)
       xmax = int(rois.getElementsByTagName("xmax")[0].firstChild.data)
       ymin = int(rois.getElementsByTagName("ymin")[0].firstChild.data)
       ymax = int(rois.getElementsByTagName("ymax")[0].firstChild.data)
       label= np.zeros((2047,1520))
       for i in range(xmin,xmax):
          for j in range(ymin,ymax):
             label[j][i]=1
       labelList.append(label)
  return np.array(imgList),np.array(labelList)

def load_val():
  imgList=[]
  labelList=[]
  files = os.listdir('D:/most/processedXR/side/predict/') #test path
  for file in files:
     if file.endswith(".xml"):continue
     fi_d = os.path.join('D:/most/processedXR/side/predict/',file) # test path
     img=Image.open(fi_d)
     img=img.crop((0,0,1520,2047))
     im=np.array(img.resize((1520,2047), Image.ANTIALIAS))
     imgList.append(im)
     filename=file.split('.')[0]
     dom = xmldom.parse('D:/most/processedXR/side/predict/'+filename+'.xml') #test path
     rois=dom.documentElement
     xmin = int(rois.getElementsByTagName("xmin")[0].firstChild.data)
     xmax = int(rois.getElementsByTagName("xmax")[0].firstChild.data)
     ymin = int(rois.getElementsByTagName("ymin")[0].firstChild.data)
     ymax = int(rois.getElementsByTagName("ymax")[0].firstChild.data)
     label= np.zeros((2047,1520))
     for i in range(xmin,xmax):
          for j in range(ymin,ymax):
             label[j][i]=1
     labelList.append(label)
     return np.array(imgList),np.array(labelList)
    
if __name__ == '__main__':
    parameters = [sherpa.Ordinal(name='batch_size', range=[16,32,64,128,256]),
                  sherpa.Ordinal(name='steps', range=[16,32,64,128,256]),
                  sherpa.Ordinal(name='epoch', range=[4096])]
    alg = sherpa.algorithms.BayesianOptimization(max_num_trials=500)
    study = sherpa.core.Study(parameters = parameters,
                         algorithm = alg,
                         lower_is_better = False,
                         disable_dashboard=True)
  
    for trial in study:
          model = Sequential()
          model.add(Convolution2D(filters=32, kernel_size =(3, 3), border_mode='same', input_shape=(2047,1520,1))) #if the memory is small, just try what we did for U-NET.
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(MaxPooling2D(pool_size=(2,2)))
          model.add(Convolution2D(filters=32, kernel_size =(3, 3), border_mode='same'))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(Convolution2D(filters=32, kernel_size =(3, 3), border_mode='same'))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(MaxPooling2D(pool_size=(2,2)))
          model.add(Convolution2D(filters=64, kernel_size =(3, 3), border_mode='same'))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(Convolution2D(filters=64, kernel_size =(3, 3), border_mode='same'))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(MaxPooling2D(pool_size=(2,2)))
          model.add(Convolution2D(filters=96, kernel_size =(3, 3), border_mode='same'))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(Convolution2D(filters=96, kernel_size =(3, 3), border_mode='same'))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          model.add(UpSampling2D(size=(8, 8),data_format="channels_last"))
          model.add(Convolution2D(filters=1,kernel_size =(1, 1), border_mode='same'))
          model.add(Activation('sigmoid'))
          model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

          datagen = ImageDataGenerator(
                  featurewise_center=False, 
                  samplewise_center=True,  # set each sample mean to 0
                  featurewise_std_normalization=False,  
                  samplewise_std_normalization=True) 
          X_train, Y_train = load()
          X_test, Y_test = load_val()
          X_train = X_train.reshape( len(X_train), len(X_train[0]), len(X_train[0][0]),1)
          X_test = X_test.reshape( len(X_test), len(X_test[0]), len(X_test[0][0]),1)
          Y_train = Y_train.reshape( len(Y_train),len(Y_train[0]), len(Y_train[0][0]),1)
          Y_test = Y_test.reshape( len(Y_test), len(Y_test[0]), len(Y_test[0][0]),1)
          X_train = X_train.astype('float32')
          X_test = X_test.astype('float32')
          X_train /= 255
          X_test /= 255
          datagen.fit(X_train) 
          for i in range(len(X_test)):
                X_test[i] = datagen.standardize(X_test[i])
          earlystop=EarlyStopping(monitor='val_iou_score', min_delta=0, patience=80, verbose=1, mode='max', restore_best_weights=True)
          callbacklist = [earlystop]
          history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=trial.parameters['batch_size']),steps_per_epoch=trial.parameters['steps'],epochs=trial.parameters['epoch'],shuffle=True,validation_data=(X_test, Y_test), verbose=1,callbacks=callbacklist)

          model.save("FCN_" + str(time.time())+".h5") #models are saved with time stamps
