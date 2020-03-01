from __future__ import print_function
import numpy as np
from PIL import Image
import xml.dom.minidom as xmldom
import os
import cv2 as cv

 #This file for creating label files
f=open('newsideXRJSNtrainlabel.txt','a') #label file path
files = os.listdir('/data/jiao/XR/processedXR/side/training/') #image path
reader = open("/data/jiao/JSN.csv") #original label path
data=reader.readlines()
for file in files:
     if file.endswith(".xml"):continue
     fi_d = os.path.join('/data/jiao/XR/processedXR/side/training/',file) #image path
     filename=file[0:6]
     dom = xmldom.parse('/data/jiao/XR/processedXR/side/training/'+filename+'.xml') #image path also has the labelled ROI xml files
     rois=dom.documentElement
     xmin = int(rois.getElementsByTagName("xmin")[0].firstChild.data)
     xmax = int(rois.getElementsByTagName("xmax")[0].firstChild.data)
     ymin = int(rois.getElementsByTagName("ymin")[0].firstChild.data)
     ymax = int(rois.getElementsByTagName("ymax")[0].firstChild.data)
     patient=file.split('_')[0]
     direction=file.split('_')[1].split('.')[0]
     label="E"
     for row in data:
         if patient in row.split(",")[0]:
              if direction in "L":
                 label=row.split(",")[2]
              else:
                 label=row.split(",")[4]
              break
     if "E" not in label:
        temp=int(label)
        f.write(fi_d+","+str(xmin)+','+str(ymin)+","+str(xmax)+","+str(ymax)+","+str(temp)+"\n")
f.close()
  
