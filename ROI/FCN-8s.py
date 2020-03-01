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
from keras.preprocessing.image import ImageDataGenerator
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

def crop( o1 , o2 , i  ):

	o_shape2 = Model( i  , o2 ).output_shape
	outputHeight2 = o_shape2[2]
	outputWidth2 = o_shape2[3]
	o_shape1 = Model( i  , o1 ).output_shape
	outputHeight1 = o_shape1[2]
	outputWidth1 = o_shape1[3]
	cx = abs( outputWidth1 - outputWidth2 )
	cy = abs( outputHeight2 - outputHeight1 )
	if outputWidth1 > outputWidth2:
		o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )) )(o1)
	else:
		o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o2)	
	if outputHeight1 > outputHeight2 :
		o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )))(o1)
	else:
		o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )))(o2)
	return o1 , o2 

def load():
  imgList=[]
  labelList=[]
  files = os.listdir('D:/most/processedXR/side/training/') #training images path
  shuffle(files)
  for file in files:
       if file.endswith(".xml"):continue
       fi_d = os.path.join('D:/most/processedXR/side/training/',file) #training images path
       img=Image.open(fi_d)
       img=img.crop((0,0,1520,2047))
       im=np.array(img.resize((1520,2047), Image.ANTIALIAS))
       imgList.append(im)
       filename=file.split('.')[0]
       dom = xmldom.parse("D:/most/processedXR/side/training/"+filename+".xml") #training images path, where also contains ROI labels
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
  files = os.listdir('D:/most/processedXR/side/predict/') #test image path
  for file in files:
     if file.endswith(".xml"):continue
     fi_d = os.path.join('D:/most/processedXR/side/predict/',file) #test image path
     img=Image.open(fi_d)
     img=img.crop((0,0,1520,2047))
     im=np.array(img.resize((1520,2047), Image.ANTIALIAS))
     imgList.append(im)
     filename=file.split('.')[0]
     dom = xmldom.parse('D:/most/processedXR/side/predict/'+filename+'.xml') #test images path, where also contains ROI labels
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
      
batch_size=128
steps=128

img_input = Input(shape=(2047,1520,1)) #if memory is not allowed running, just try what we did for U-NET
x=Convolution2D(filters=32, kernel_size =(3, 3),activation='relu',border_mode='same', input_shape=(2047,1520,1))(img_input)
x=BatchNormalization()(x)
x=MaxPooling2D(pool_size=(2,2))(x)
f1=x
x=Convolution2D(filters=32, kernel_size =(3, 3),activation='relu',border_mode='same')(x)
x=BatchNormalization()(x)
x=Convolution2D(filters=32, kernel_size =(3, 3),activation='relu',border_mode='same')(x)
x=BatchNormalization()(x)
x=MaxPooling2D(pool_size=(2,2))(x)
f2=x
x=Convolution2D(filters=64, kernel_size =(3, 3),activation='relu',border_mode='same')(x)
x=BatchNormalization()(x)
x=Convolution2D(filters=64, kernel_size =(3, 3),activation='relu',border_mode='same')(x)
x=BatchNormalization()(x)
x=MaxPooling2D(pool_size=(2,2))(x)
f3=x
x=Convolution2D(filters=96, kernel_size =(3, 3),activation='relu',border_mode='same')(x)
x=BatchNormalization()(x)
x=Convolution2D(filters=96, kernel_size =(3, 3),activation='relu',border_mode='same')(x)
x=BatchNormalization()(x)
x=MaxPooling2D(pool_size=(2,2))(x)
f4=x
o = f4
o = UpSampling2D(size=(2, 2),data_format="channels_last")(o)
o = Conv2D(1,  ( 1 , 1 ) ,kernel_initializer='he_normal')(o)
o = Activation('sigmoid')(o)
o2 = f3
o2 = Conv2D(1 ,  ( 1 , 1 ) ,kernel_initializer='he_normal')(o2)
o , o2 = crop( o , o2 , img_input)
o = Add()([ o , o2 ])
o = UpSampling2D(size=(2, 2),data_format="channels_last")(o)
o = Conv2D(1,  ( 1 , 1 ) ,kernel_initializer='he_normal')(o)
o = Activation('sigmoid')(o)
o2 = f2 
o2 = Conv2D( 1,  ( 1 , 1 ) ,kernel_initializer='he_normal')(o2)
o2 , o = crop( o2 , o , img_input )
o  = Add()([ o2 , o ])
o = UpSampling2D(size=(4, 4),data_format="channels_last")(o)
o = Conv2D(1,  ( 1 , 1 ) ,kernel_initializer='he_normal')(o)
o = Activation('sigmoid')(o)
model = Model( img_input , o )
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
earlystop=EarlyStopping(monitor='val_iou_score', min_delta=0, patience=200, verbose=1, mode='max', restore_best_weights=True)
callbacklist = [earlystop]
history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),steps_per_epoch=steps,epochs=2000,shuffle=True,validation_data=(X_test, Y_test), verbose=1,callbacks=callbacklist)
score, acc = model.evaluate(X_test,Y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save("FCN_" + str(time.time())+".h5") #models are saved with time stamps
