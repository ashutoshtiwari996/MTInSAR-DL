#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:17:08 2023

@author: ashutosh
"""


'''
# A program for creating elite pixel labels in multi-temporal InSAR

# Input: A stack of interferograms exported from ESA SNAP software 
# in Bigtiff/Geotiff format (exported from matlab) or in tiff format (exported from ESA SNAP software). 

# Output: 
    Input data in matlab/python supported format
    A map with labels 0 and 1, 0 denoting non-PS pixels, and 1 denoting PS pixels 

'''

"""
Packages required

numpy scipy mat73 rasterio hdf5storage matplotlib patchify PIL

"""

#Hare Krishna

# %Hare Krishna

# % Program to create labelled output

import rasterio as rs
import numpy as np
import scipy.io as spio
import pandas
import hdf5storage as hs
import mat73
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from patchify import patchify, unpatchify
from PIL import Image


fpath='/home/ashutosh/JSG29-DS and ML in Geodesy/Datasets/MNNIT'

dataset=rs.open(fpath+'/subset_4_of_S1A_IW_SLC__1SDV_20211224T124730_20211224T124756_041151_04E3C1_90AE_split_Orb_Stack_deb_ifg_Cnv.tif') #Your interferometric stack here 

#reading specific bands from data

dataset.meta


# Iterating over range 0 to n-1
 
n=56 #no. of interferograms 

phase_bands=np.zeros((n, dataset.height, dataset.width))

# ph_size=np.zeros((n,1))

# ph_size[0]=4

phase_bands[0,:,:]=dataset.read(4)

print(np.count_nonzero(phase_bands))



for i in range(1, n-1):
    # phase_bands[:,:,i]=(i+1)*5+2
    phase_bands[i,:,:]=dataset.read(((i+1)*5+2))
  

X=phase_bands  #input image stack 


ppath='/home/ashutosh/JSG29-DS and ML in Geodesy/Datasets/MNNIT/StaMPS_export';


patch1=spio.loadmat(ppath+'/PATCH_1/ps2.mat');

ij1=patch1['ij'];

lonlat1=patch1['lonlat'];


patch2=spio.loadmat(ppath+'/PATCH_2/ps2.mat');

ij2=patch2['ij'];

lonlat2=patch2['lonlat'];


patch3=spio.loadmat(ppath+'/PATCH_3/ps2.mat');

ij3=patch3['ij'];

lonlat3=patch3['lonlat'];


patch4=spio.loadmat(ppath+'/PATCH_4/ps2.mat');

ij4=patch4['ij'];

lonlat4=patch4['lonlat'];



# %repeat if more patches are used in processing


ij_cand=np.row_stack((ij1,ij2,ij3,ij4))

lonlat_cand=np.row_stack((lonlat1, lonlat2, lonlat3, lonlat4))


# %reading information for finally selected PS pixels


patch=spio.loadmat(ppath+ '/ps2.mat') #

ij_final=patch['ij'];

lonlat_final=patch['lonlat'];


del patch1,patch2, patch3, patch4, patch

# %now use this to generate ps labels from the final lonlat and ij

# % Jai Shri RadheShyam

ps_label=np.zeros((ij_cand.shape[0],1));

ij2_mat=np.zeros((phase_bands.shape[1],phase_bands.shape[2]))


for i in range (0, ij_final.shape[0]-1):
    ij2_mat[ij_final[i,1],ij_final[i,2]]=1;

y=ij2_mat

from matplotlib import pyplot as plt
plt.imshow(y, interpolation='nearest')
plt.show()

#Another display

from scipy.misc import imshow
imshow(y)

spio.savemat(fpath+'/MTInSAR_training_dataset.mat', dict(X=X, y=y))

#Hare Krishna

#crop input to modulo size
