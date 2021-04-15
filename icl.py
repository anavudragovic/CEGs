#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:51:06 2021

@author: anavudragovic
"""

import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
#from astropy.visualization import SqrtStretch
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.visualization import simple_norm
from photutils.segmentation import make_source_mask
#from photutils.datasets import make_100gaussians_image
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
import numpy.ma as ma

#data = make_100gaussians_image()
path = "/Users/anavudragovic/vidojevica/compact_ellipticals_at_Milankovic"
galfile="ngc12710-2.fits"
os.chdir(path)

galimg = fits.open(galfile)
hdu = galimg[0]
data = hdu.data
wcs = WCS(hdu.header) 

mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print((mean, median, std)) 
mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=11)
mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
print((mean, median, std)) 


# Estimate background
sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()


norm = simple_norm(data*~mask, 'sqrt', percent=99.)
fig = plt.figure(figsize=(8,8))
#plt.title('dilate='+str(stdev))
im1=plt.imshow(data*~mask, origin='lower', cmap='Greys_r', norm=norm)
plt.colorbar(im1)
plt.savefig('data_masked.png')
#plt.savefig('dilate'+str(stdev)+'.png')
# Save extended mask image:
#hdu.header.update(wcs.to_header())
#hdu.data = segmap_gauss
#hdu.writeto("seg_dilate"+str(stdev)+".fits", overwrite=True)

# Make another mask for blank areas in the original image 
bkgmask = (data == 0.0)   

# Create a Background2D object using a box size of mesh_size x mesh_size and a 5x5 median filter
# 50: 1 100:7 150:10 200:15 300:25 500:26
mesh_size=50
bkg = Background2D(data, (mesh_size, mesh_size), filter_size=(3, 3), mask=mask, coverage_mask=bkgmask,sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,exclude_percentile=1,fill_value=0.0)
print(bkg.background_median, bkg.background_rms_median)  
fig = plt.figure(figsize=(12, 4))
plt.subplots_adjust(hspace=0., wspace=0.,left=None, right=None, bottom=None, top=None)
plt.subplot(1, 3, 1)
# Background image 
plt.xlim(250,3000)
plt.ylim(1500,3500)
plt.axis('off')
im1=plt.imshow(bkg.background, origin='lower', cmap='Greys_r')
#plt.colorbar(im1,cmap='Greys_r')

plt.subplot(1, 3, 2)
#plt.title(str(mesh_size)+" pix")
# Original image with meshes
plt.xlim(250,3000)
plt.ylim(1500,3500)
plt.axis('off')
im2=plt.imshow(data*~mask, norm=simple_norm(data,'sqrt',percent=90.), origin='lower',
            cmap='Greys_r', interpolation='nearest')
#plt.colorbar(im2,cmap='Greys_r')
bkg.plot_meshes(outlines=True, color='#1f77b4')
plt.subplot(1, 3, 3)    
plt.xlim(250,3000)
plt.ylim(1500,3500)
plt.axis('off')
#im3=plt.imshow(gaussian_filter(data-bkg.background,sigma=7), norm=simple_norm(data,'sqrt',percent=90.), origin='lower',cmap='Greys_r', interpolation='nearest')
im3=plt.imshow(data-bkg.background, norm=simple_norm(data,'sqrt',percent=90.), origin='lower',cmap='Greys_r', interpolation='nearest')

#plt.colorbar(im3,cmap='Greys_r')
plt.savefig("icl"+str(mesh_size)+"_zoom.png")

# Save bkg subtracted fits image:
hdu.header.update(wcs.to_header())
hdu.data = data - bkg.background # save for image processing
hdu.writeto("iclsub"+str(mesh_size)+"_" + galfile, overwrite=True)

# Save background image:
hdu.header.update(wcs.to_header())
hdu.data = bkg.background
hdu.writeto("iclbkg"+str(mesh_size)+"_" + galfile, overwrite=True)

print(mesh_size," : ",np.mean(bkg.background*~bkgmask),np.std(bkg.background*~bkgmask))
#im=plt.imshow(mask,cmap='Greys_r',origin="lower")
#plt.colorbar(im)

# 
#import imexam
#ds9=imexam.connect(target="",path="/Applications/SAOImageDS9.app/Contents/MacOS/ds9",viewer="ds9",wait_time=10)
#ds9.load_fits(galfile)
#ds9.scale()
#ds9.exam.radial_profile_pars['rplot'][0] = 240
#ds9.exam.curve_of_growth_pars['rplot'][0]=250
#ds9.imexam()
