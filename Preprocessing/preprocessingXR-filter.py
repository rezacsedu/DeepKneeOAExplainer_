import numpy as np
from numpy import *
import os
import optparse
from PIL import Image
import png
import pylab
from scipy import misc, ndimage
from medpy.filter.smoothing import anisotropic_diffusion
from sklearn import preprocessing

def histogram_t(tb):
    totalpixel=0    
    maptb=[]        
    count=len(tb)
    for i in range(count):
        totalpixel+=tb[i]
        maptb.append(totalpixel)

    for i in range(count):
        maptb[i]=int(round((maptb[i]*(count-1))/totalpixel))
   
    def histogram(light):
        return maptb[light]
    return histogram

imagepath='T:/CleanedMostData/XR/cleanedXR/' #image folder path, please pay attention to here images are already renamed with format "patient_direction". This step is finished by data format transformation from DICOM to .PNG.
files = os.listdir(imagepath)
for fi in files:
  fi_d = os.path.join(imagepath,fi)
  img=Image.open(fi_d)
  if 'P' in fi:  #for coronal radiographs
    (tempx1,tempy1)=img.size #original image separation
    width=tempx1//2
    left=img.crop((0,0,width,tempy1))
    right=img.crop((width,0,tempx1,tempy1))
    imgl = Image.new('RGBA',(width,tempy1))
    imgr = Image.new('RGBA',(width,tempy1))
    imgl.paste(left)
    imgr.paste(right)
    imgr=imgr.transpose(Image.FLIP_LEFT_RIGHT) #image flipping
    outl=imgl.resize((1023,2047),Image.ANTIALIAS) #image resize
    outr=imgr.resize((1023,2047),Image.ANTIALIAS)
    hisl=outl.histogram() #histogram
    hisfuncl=histogram_t(hisl) 
    iml=outl.point(hisfuncl)   
    hisr=outr.histogram() 
    hisfuncr=histogram_t(hisr) 
    imr=outr.point(hisfuncr)    
    ir = anisotropic_diffusion(np.array(imr)) #noise removal
    il = anisotropic_diffusion(np.array(iml))
    temp='T:/CleanedMostData/XR/normXR/front/'+fi[0:4] #processed image path for saving
    imagel = Image.fromarray(il.astype('uint8')).convert("L")
    imager = Image.fromarray(ir.astype('uint8')).convert("L")    
    imager.save(temp+'_PR.png')
    imagel.save(temp+'_PL.png')
  else: #for sagittal radiographs
    out=img.resize((1520,2047),Image.ANTIALIAS)
    his=out.histogram() 
    hisfunc=histogram_t(his) 
    im=out.point(hisfunc)    
    i = anisotropic_diffusion(np.array(im))
    image = Image.fromarray(i.astype('uint8')).convert("L")
    image.save(os.path.join('T:/CleanedMostData/XR/normXR/side/',fi)) #processed image path for saving
