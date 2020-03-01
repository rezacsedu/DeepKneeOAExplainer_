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
import tensorflow as tf
import keras
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def get_session(): 
  config = tf.ConfigProto() 
  config.gpu_options.allow_growth = True 
  return tf.Session(config=config) 
# use this environment flag to change which GPU to use 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# set the modified tf session as backend in keras 
keras.backend.tensorflow_backend.set_session(get_session())

trainingpath='D:/most/processedXR/front/training/'
testpath='D:/most/processedXR/front/predict/'

def load():
  imgList=[]
  labelList=[]
  files = os.listdir()
  shuffle(files)
  for file in files:
       if file.endswith(".xml"):continue
       fi_d = os.path.join(trainingpath,file)
       img=Image.open(fi_d)
       img=img.crop((0,0,1023,2047)) #due to the memory problem, I resize images with 1/8 size. If you have enough GPU memeory, you dont need that.
       im=np.array(img.resize((128,256), Image.ANTIALIAS))
       imgList.append(im)
       filename=file.split('.')[0]
       dom = xmldom.parse(trainingpath+filename+".xml")
       rois=dom.documentElement
       xmin = int(int(rois.getElementsByTagName("xmin")[0].firstChild.data)/8)
       xmax = int(int(rois.getElementsByTagName("xmax")[0].firstChild.data)/8)
       ymin = int(int(rois.getElementsByTagName("ymin")[0].firstChild.data)/8)
       ymax = int(int(rois.getElementsByTagName("ymax")[0].firstChild.data)/8)
       label= np.zeros((256,128))
       for i in range(xmin,xmax):
          for j in range(ymin,ymax):
             label[j][i]=1
       labelList.append(label)
  return np.array(imgList),np.array(labelList)

def load_val():
  imgList=[]
  labelList=[]
  files = os.listdir()
  for file in files:
     if file.endswith(".xml"):continue
     fi_d = os.path.join(testpath,file)
     img=Image.open(fi_d)
     img=img.crop((0,0,1023,2047)) #due to the memory problem, I resize images with 1/8 size. If you have enough GPU memeory, you dont need that.
     im=np.array(img.resize((128,256), Image.ANTIALIAS))
     imgList.append(im)
     filename=file.split('.')[0]
     dom = xmldom.parse(testpath+filename+'.xml')
     rois=dom.documentElement
     xmin = int(int(rois.getElementsByTagName("xmin")[0].firstChild.data)/8)
     xmax = int(int(rois.getElementsByTagName("xmax")[0].firstChild.data)/8)
     ymin = int(int(rois.getElementsByTagName("ymin")[0].firstChild.data)/8)
     ymax = int(int(rois.getElementsByTagName("ymax")[0].firstChild.data)/8)
     label= np.zeros((256,128))
     for i in range(xmin,xmax):
          for j in range(ymin,ymax):
             label[j][i]=1
     labelList.append(label)
     return np.array(imgList),np.array(labelList)

BACKBONE = 'resnet18' #Here you can choose backbone 
preprocess_input = get_preprocessing(BACKBONE)      

batch_size=80

model = Unet(backbone_name='resnet18', encoder_weights=None, input_shape=(None, None, 1)) #You even can choose segmentation architectures, the detail choices you can check https://segmentation-models.readthedocs.io/en/latest/
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
callbacks_list = [earlystop]
history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),steps_per_epoch=32,epochs=1000,validation_data=(X_test,Y_test),callbacks=callbacks_list,verbose=1)
score, acc = model.evaluate(X_test,Y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save("UNET_" + str(time.time())+".h5") #model are saved with time stamps
print("Model saved")



