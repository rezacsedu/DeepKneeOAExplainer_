import numpy as np
import cv2
import os
import optparse
from PIL import Image
import png
import pylab
from scipy import misc, ndimage
from medpy.filter.smoothing import anisotropic_diffusion
from PIL import ImageFilter
   
patients = os.listdir('T:/CleanedMostData/MRI/temp/') #image path, where they keep the file structure from MOST
for p in patients:
   p_d = os.path.join('T:/CleanedMostData/MRI/temp/',p)            
   directions = os.listdir(p_d)                
   for d in directions:
      d_d = os.path.join(p_d,d)
      stages=os.listdir(d_d)
      for s in stages:
          s_d=os.path.join(d_d,s)
          dates=os.listdir(s_d)
          for da in dates:
              da_d=os.path.join(s_d,da)
              series=os.listdir(da_d)
              for se in series:
                  se_d=os.path.join(da_d,se)
                  perspectives=os.listdir(se_d)
                  for pe in perspectives:
                     if "SER1" in pe or "SER2" in pe: #1 is for sagittal view, 2 is for axial view, if you need more, check SER3 and SER4.
                        pe_d=os.path.join(se_d,pe)
                        im=Image.open(os.path.join(pe_d,'avreage.png'))
                        print(pe_d)
                        img=np.array(im)
                        if d in "R":
                           img=np.fliplr(img)
                        i = anisotropic_diffusion(img,niter=5,kappa=20)
                        image = Image.fromarray(i.astype('uint8')).convert("L")
                        #image=image.filter(ImageFilter.SHARPEN)
                        image=image.filter(ImageFilter.EDGE_ENHANCE)
                        if d in "R":
                           if "SER1" in pe:
                              image.save('T:/CleanedMostData/MRI/processedMRI/side/'+p[1:5]+"_R.png") #here they change the storage structure
                           if "SER2" in pe:
                              image.save('T:/CleanedMostData/MRI/processedMRI/up/'+p[1:5]+"_R.png")
                        if d in "L":
                           if "SER1" in pe:
                              image.save('T:/CleanedMostData/MRI/processedMRI/side/'+p[1:5]+"_L.png")
                           if "SER2" in pe:
                              image.save('T:/CleanedMostData/MRI/processedMRI/up/'+p[1:5]+"_L.png")
                           
