from __future__ import print_function
import numpy as np
import time
import csv
import os
from sklearn.metrics import classification_report
import sklearn.metrics as sklm
import math
from scipy import stats
  
def dense_to_one_hot(labels_dense,num_clases=5):
  return np.eye(num_clases)[labels_dense]

def load_va(): #compute average softmax possibilities
  imlist=[]
  labelList=[]
  nameList=[]
  reader = open("/data/jiao/integration/final.csv") #prediction possibility file path
  data=reader.readlines()
  label="q"
  for row in data:
     for i in range(4):
        imlist.append(float(row.split(",")[i+1])+float(row.split(",")[i+5])+float(row.split(",")[i+9])+float(row.split(",")[i+13])+float(row.split(",")[i+33]))
     label=imlist.index(max(imlist))
     imlist=[]
     labelList.append(label)
     nameList.append(row.split(",")[0])
  return np.array(labelList),np.array(nameList)


def load_valY(): #read original label file
  imgList=[]
  labelList=[]
  nameList=[]
  reader = open("/data/jiao/newlabel.csv") #original label file path
  data=reader.readlines()
  files = os.listdir('/data/jiao/XR/balancedXR/front/validation/') #test set image path
  for file in files:
        if file.endswith(".xml"):continue
        patient=file.split('_')[0]
        direction=file.split('_')[1].split('.')[0]
        naming=file.split('.')[0]
        label="q"
        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label=row.split(",")[3]
              else:
                 label=row.split(",")[6]
              break
        if "V" in file:
                       label="3"
        if "8" not in label and "9" not in label and "X" not in label and '.' not in label:
          #if "." in label:
            #label='4'
          labelList.append(int(label))
          nameList.append(naming)
  return np.array(labelList),np.array(nameList)

Y_true,N_true=load_valY()
Y_pre,N_pre=load_va()
true=[]
pre=[]
for i in range(len(Y_true)-1):
    for j in range(len(Y_pre)-1):   
        if N_true[i]==N_pre[j]:
          true.append(Y_true[i])
          pre.append(Y_pre[j])
          break

print(sklm.accuracy_score(true,pre))
print(sklm.classification_report(true,pre))
print(sklm.confusion_matrix(true,pre))
print(sklm.mean_squared_error(true,pre))