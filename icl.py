#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:40:59 2021

@author: anavudragovic
"""

import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import SqrtStretch
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.visualization import simple_norm
from photutils.datasets import make_100gaussians_image
import numpy as np
from scipy.ndimage import gaussian_filter

path = "/Users/anavudragovic/vidojevica/compact_ellipticals_at_Milankovic"
galfile="ngc12710-2.fits"
segfile="ngc12710-2.seg.fits"
os.chdir(path)

galimg = fits.open(galfile)
hdu = galimg[0]
data = hdu.data
wcs = WCS(hdu.header) 
segimg = fits.open(segfile)
seg = segimg[0].data

norm = simple_norm(data, 'sqrt', percent=99.)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(data, origin='lower', norm=norm,cmap='Greys',interpolation='nearest')
fig.colorbar(im)
# Dilate mask with gaussian filter to expand it around objects
segmap = np.nan_to_num(seg)
segmap[segmap > 0.0]=100
segmap[segmap <= 0.0] = 0
segmap_gauss=gaussian_filter(segmap,sigma=5)
# Make another mask for blank areas in the original image 
bkgmask = (data != 0.0)   
# Merge two masks into the single one     
mask = ((~bkgmask) | (segmap_gauss > 0))
# Estimate background
sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
# Create a Background2D object using a box size of mesh_size x mesh_size and a 3x3 median filter
mesh_size=300
bkg = Background2D(data, (mesh_size, mesh_size), filter_size=(3, 3), mask=mask, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,exclude_percentile=30)
fig = plt.figure(figsize=(8, 4))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.subplot(1, 2, 1)
# Background image multiplied with mask to show excluded areas/objects
im1=plt.imshow(bkg.background*(~mask), origin='lower', cmap='Greys_r')
plt.colorbar(im1,cmap='Greys_r')

plt.subplot(1, 2, 2)
plt.title(str(mesh_size)+" pix")
# Final, background subtracted image
im2=plt.imshow((data - bkg.background)*bkgmask, norm=norm, origin='lower',
            cmap='Greys_r', interpolation='nearest')
plt.colorbar(im2,cmap='Greys_r')
bkg.plot_meshes(outlines=True, color='#1f77b4')
plt.savefig("icl"+str(mesh_size)+".png")

# Save bkg subtracted fits image:
hdu.header.update(wcs.to_header())
# Here bkg subtracted image is multipled with bkgmask (== mask of empty areas)
hdu.data = (data - bkg.background)*bkgmask # save for image processing
hdu.writeto("iclsub"+str(mesh_size)+"_" + galfile, overwrite=True)
