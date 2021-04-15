#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:54:56 2021

@author: anavudragovic


Isophote analysis of NGC1270 galaxy. Two type of images can be given as input:
    (1) ICL background subtracted image
    (2) original image (fully reduced)

In the (1)st case, different input images should be fed to the script: 
iclbkg*_ngc12710-2.fits, where * = mesh_size
Output: mag_iclbkg*_ngc12710-2.txt with ellipse params

In the (2)nd case, ngc12710-2.fits itself is processed in three ways:
    (2.1) label='_base' refers to the image without any bkg subtraction
    (2.2*) label='_bkg' refers to the image where bkg is measured and subtracted
    (2.3*) label='_bkg1m' refers to the image where mean bkg and stdev are measured and
        background subtracted is bkg-stdev
    (2.4*) label='_bkg1p' refers to the image where mean bkg and stdev are measured and
        background subtracted is bkg+stdev
(*) background is measured as the mean value of 20 boxes 10x10 pix large placed
    around galaxy at the distance of 3*major_axis ~ 180pix =70 arcsec 
Output: mag_ngc12710-2_label.txt with ellipse params
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.stats import sigma_clipped_stats

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture, RectangularAperture, PixelAperture
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model
from photutils.aperture import aperture_photometry
from scipy.ndimage import gaussian_filter
from photutils.aperture import SkyCircularAperture
#from photutils.segmentation import detect_threshold, detect_sources
from photutils.morphology import data_properties
from photutils.datasets import make_4gaussians_image
from copy import deepcopy
import scipy as sp
import scipy.optimize as optimize
import numpy.ma as ma

def angles_in_ellipse(num, a, b):
    #assert num > 0,'N>0'
    #assert a < b,'a>b'
    print('a = ',a, ' b= ',b)
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e = (1.0 - b ** 2.0 / a ** 2.0) ** 0.5
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(
            lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x 
    return angles


path = "/Users/anavudragovic/vidojevica/compact_ellipticals_at_Milankovic"
#imfile = "iclsub100_ngc12710-2.fits"
imfile = "iclsub100_ngc12710-2.fits"
segfile = "ngc12710-2.seg.fits"
os.chdir(path)
label=''
base=os.path.basename(imfile)
out=os.path.splitext(base)[0]
galimg = fits.open(imfile)
galaxy = galimg[0].data
hdu = galimg[0]
wcs = WCS(hdu.header) 
segimg = fits.open(segfile)
seg = segimg[0].data
seghdu = segimg[0]
segwcs = WCS(seghdu.header) 
#seg[seg > 0.0]=1 # All detected objects set to 1

gal_name = 'NGC1270'
#ra_sat = 49.742267; dec_sat = 41.470042
center = SkyCoord.from_name(gal_name)#get_icrs_coordinates(name)
# Find the object in the segmentation image to mask it
seg_position = skycoord_to_pixel(center, segwcs)
seg_center = int(np.round(seg_position[1])),int(np.round(seg_position[0]))
num = seg[seg_center]
msk = deepcopy(seg)
box = deepcopy(seg)
#msk[msk!=num] = 1
msk[msk==num] = 0
seg[seg!=num] = 0
# Dilate mask with gaussian filter to expand it around objects
segmap = np.nan_to_num(msk)
#segmap[segmap > 0.0]=100
segmap[segmap <= 0.0] = 0
#segmap_gauss=gaussian_filter(segmap,sigma=5)
segmap_gauss = segmap

# Find celestial coordinates in the image (in pixels)
position = skycoord_to_pixel(center, wcs)
# Crop the image to smaller dimensions using celestial coordinates
# dimensions are set by the radius in pixels = 701
nsize = 701
size =  nsize * u.pixel
#aperture = SkyCircularAperture(center, r=4. * u.arcsec)
#pix_aperture = aperture.to_pixel(wcs)

print("position: ", position)
gal_cutout = Cutout2D(galaxy, position, size, wcs=wcs)
msk_cutout = Cutout2D(segmap_gauss, position, size, wcs=segwcs)
seg_cutout = Cutout2D(seg, position, size, wcs=segwcs)
box_cutout = Cutout2D(box, position, size, wcs=segwcs)
box_cutout.data[box_cutout.data>0]=1
box_cutout.data[gal_cutout.data<0]=1
data = gal_cutout.data
segmask = seg_cutout.data
mask = msk_cutout.data
#mask[data<0]=np.max(mask)+100
#data[data<0]=np.nan
norm_gal = simple_norm(data, 'sqrt', percent=99.)
norm_msk = simple_norm(mask, 'sqrt', percent=99.)
plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace=0.35, wspace=0.35)
plt.subplot(2, 2, 1)
plt.title("Image: "+str(os.path.basename(imfile)))
plt.imshow(data, norm=norm_gal, origin="lower")
plt.subplot(2, 2, 2)
plt.title("Mask: "+str(os.path.basename(segfile)))
imm=plt.imshow(mask, norm=norm_msk, origin="lower")
#plt.colorbar(imm)

# Target galaxy is the only object in segmask*data image
# It's basic geom properties are measured with data_properties
# This results in one elliptical aperture for the target galaxy
# Geompars will be passed to the ellipse instance
cat = data_properties(segmask*data)
columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
           'semiminor_sigma', 'orientation']
tbl = cat.to_table(columns=columns)
r = 2.5  # approximate isophotal extent
positioncen = (cat.xcentroid, cat.ycentroid)
a = cat.semimajor_sigma.value * r
b = cat.semiminor_sigma.value * r
theta = cat.orientation.to(u.rad).value
aperture = EllipticalAperture(positioncen, a, b, theta=theta)


# ----------------------------------------------------------------------------
# This part should be used only with original image, not icl subtracted one
# It measures the bkg inside inside 20 boxes (10x10 pix) equally spaced along the ellipse
# Inside boxes detected objects are masked with boolmask (created by Sextractor)
# Ellipse major axis for bkg is eq to 3 * major axis of the object 
# It's best to be taken as ~ 1.2 R from RC3 catalog and not simply 3*major_axis
# background = mean +/- stdev
# RC3: R_25 ~ 234 arcsec
Rc=234
nn = Rc/a
nna = nn*a
nnb = nn*b 
phi = angles_in_ellipse(40, nna, nnb)
#print(np.round(np.rad2deg(phi), 2))

e = (1.0 - nnb ** 2.0 / nna ** 2.0) ** 0.5
arcs = sp.special.ellipeinc(phi, e)
#print(np.round(np.diff(arcs), 4))
x = cat.xcentroid + nnb * np.sin(phi)
y = cat.ycentroid + nna * np.cos(phi)
box = RectangularAperture(zip(x,y), 32, 32)
boolmask = box_cutout.data > 0
phot = aperture_photometry(data, box, mask=boolmask)
bkg_mean = np.mean(phot['aperture_sum'] / box.area) # 14.985152614520166
bkg_rms = np.std(phot['aperture_sum'] / box.area) # 4.568966255984001
plt.figure(figsize=(8, 8))
norm = simple_norm(data*~boolmask, 'sqrt', percent=99.)
im=plt.imshow(data*~boolmask, cmap='viridis',interpolation='nearest',norm=norm)
plt.colorbar(im)
aperture.plot(color='#d62728')
box.plot(color='#d62728')
print(bkg_mean,bkg_rms,bkg_mean - 1*bkg_rms,bkg_mean + 1*bkg_rms,bkg_mean - 3*bkg_rms,bkg_mean + 3*bkg_rms)
#data = data - (bkg_mean+3*bkg_rms)#(bkg_mean - 1*bkg_rms) 
### np.median(data[box_cutout.data==0]) # == 15.167292
### np.std(data[box_cutout.data==0]) #== 10.078673
# ----------------------------------------------------------------------------
#datamask = deepcopy(data)
#datamask[mask>0]=0

geometry = EllipseGeometry(x0=cat.xcentroid, y0=cat.ycentroid, sma=a, eps=1-b/a,
                            pa=(theta+90)*np.pi/180.)


ellipse = Ellipse(ma.masked_where(mask > 0, data), geometry)
#isolist = ellipse.fit_image(integrmode='median', step=5, linear=True, 
#                                  maxsma=nsize/2, fflag=0.3, sclip=3, nclip=3)
isolist = ellipse.fit_image(integrmode='median', step=.1, linear=False, 
                                  maxsma=nsize/2, fflag=0.3, sclip=3, nclip=3)


# Try to fit further more (from the last ellipse) with larger step to increase S/N
# and capture outer parts of the galaxy
#geometry = EllipseGeometry(x0=cat.xcentroid, y0=cat.ycentroid, sma=np.max(isolist.sma), 
#                           eps=1-b/a, pa=(theta+90)*np.pi/180.)
#ellipse = Ellipse(data, geometry)
#isolist_outer = ellipse.fit_image(integrmode='median', step=0.3, minsma=isolist.sma[-1],
#                                  maxsma=nsize/2, fflag=0.3, sclip=3.0, nclip=3)

# Join two list excluding the last solution from the outer fit since stop_code=4    
#isolist_wide = isolist + isolist_outer[2:-1]
isolist_wide=isolist
# Make a bmodel and a residual image
model_image = build_ellipse_model(data.shape, isolist_wide)
residual = data - model_image

flux_err = np.sqrt(isolist_wide.intens + isolist_wide.pix_stddev**2)
mag_err = 1.0857 * isolist_wide.int_err / isolist_wide.intens
mag = -2.5*np.log10(isolist_wide.intens)

# ----------------------------------------------------------------------------
#                  ***      W R I T E    R E S U L T S     ***
# ----------------------------------------------------------------------------
t = isolist_wide.to_table()
t.add_column(mag, name='mag')
t.add_column(mag_err, name='mag_err')
t.write('mag_'+out+label+'.txt', format='ascii', overwrite=True)
# ----------------------------------------------------------------------------
#                  ***        P L O T    R E S U L T S      ***
# ----------------------------------------------------------------------------
# Plot ellipses over galaxy image; add bmodel and residual image
plt.figure(figsize=(8, 8))
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 3), nrows=1, ncols=3)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
ax1.imshow(data, origin='lower', norm=norm_gal, cmap='viridis')
ax1.set_title('Data')

smas = np.linspace(np.min(isolist_wide.sma), np.max(isolist_wide.sma), 30)
for sma in smas:
    iso = isolist_wide.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax1.plot(x, y, color='white', lw=0.5,ls='--')

ax2.imshow(model_image, origin='lower',norm=norm_gal, cmap='viridis')
ax2.set_title('Ellipse Model')

ax3.imshow(residual, origin='lower',norm=norm_gal, cmap='viridis')
ax3.set_title('Residual')
plt.savefig('model_'+out+label+'.png')

# Plote isophotes from joined ellipse fit
plt.figure(figsize=(8, 8))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(data, vmin=0, vmax=1200)
ax.set_title(gal_name)

isos = []
smas = np.linspace(np.min(isolist_wide.sma), np.max(isolist_wide.sma), 30)
for sma in smas:
    iso = isolist_wide.get_closest(sma)
    isos.append(iso)
    x, y, = iso.sampled_coordinates()
    plt.plot(x, y, color='w',lw=0.5,ls='--')
plt.savefig('isophotes_'+out+label+'.png')


# Plot basic geom parameters
plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace=0.35, wspace=0.35)

llim = np.nanmin(isolist.eps) - np.nanstd(isolist.eps)/2
hlim = np.nanmax(isolist.eps) + np.nanstd(isolist.eps)/2
plt.subplot(2, 2, 1)
plt.ylim(llim,hlim)
plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
             fmt='o',  mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('Ellipticity')

llim = np.nanmin(isolist.pa/np.pi*180.) - np.nanstd(isolist.pa/np.pi*180.)/2
hlim = np.nanmax(isolist.pa/np.pi*180.) + np.nanstd(isolist.pa/np.pi*180.)/2
plt.subplot(2, 2, 2)
plt.ylim(llim,hlim)
plt.errorbar(isolist.sma, isolist.pa/np.pi*180.,
             yerr=isolist.pa_err/np.pi* 80., fmt='o', mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('PA (deg)')

llim = np.nanmin(isolist.x0) - np.nanstd(isolist.x0)/2
hlim = np.nanmax(isolist.x0) + np.nanstd(isolist.x0)/2
plt.subplot(2, 2, 3)
plt.ylim(llim,hlim)
plt.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o', mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('x0')

llim = np.nanmin(isolist.y0) - np.nanstd(isolist.y0)/2
hlim = np.nanmax(isolist.y0) + np.nanstd(isolist.y0)/2
plt.subplot(2, 2, 4)
plt.ylim(llim,hlim)
plt.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o', mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('y0')
plt.savefig('geompar_'+out+label+'.png')

# Plot higher harmonics 
plt.figure(figsize=(10, 5))
#llim = np.min(isolist_wide.a3[:-2]) - np.std(isolist_wide.a3)
#hlim = np.max(isolist_wide.a3[:-2]) + np.std(isolist_wide.a3)
llim=-0.1
hlim=0.1
plt.subplot(221)
plt.ylim(llim,hlim)
plt.errorbar(isolist_wide.sma, isolist_wide.a3, yerr=isolist_wide.a3_err, fmt='o', mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.axhline(0,ls='--',color='k',lw=2)
plt.xlabel('Semimajor axis length')
plt.ylabel('A3')

plt.subplot(222)
#llim = np.min(isolist_wide.b3[:-2]) - np.std(isolist_wide.b3)
#hlim = np.max(isolist_wide.b3[:-2]) + np.std(isolist_wide.b3)
llim=-0.1
hlim=0.1
plt.ylim(llim,hlim)
plt.errorbar(isolist_wide.sma, isolist_wide.b3, yerr=isolist_wide.b3_err, fmt='o', mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.axhline(0,ls='--',color='k')
plt.xlabel('Semimajor axis length')
plt.ylabel('B3')

plt.subplot(223)
#llim = np.min(isolist_wide.a4) - np.std(isolist_wide.a4)
#hlim = np.max(isolist_wide.a4) + np.std(isolist_wide.a4)
llim=-0.1
hlim=0.1
plt.ylim(llim,hlim)
plt.errorbar(isolist_wide.sma, isolist_wide.a4, yerr=isolist_wide.a4_err, fmt='o', mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.axhline(0,ls='--',color='k')
plt.xlabel('Semimajor axis length')
plt.ylabel('A4')

plt.subplot(224)
#llim = np.min(isolist_wide.b4[:-2]) - np.std(isolist_wide.b4)
#hlim = np.max(isolist_wide.b4[:-2]) + np.std(isolist_wide.b4)
llim=-0.1
hlim=0.1
plt.ylim(llim,hlim)
plt.errorbar(isolist_wide.sma, isolist_wide.b4, fmt='o', yerr=isolist_wide.b4_err, mfc='white',mec='k',ecolor='k',alpha=0.7)
plt.axhline(0,ls='--',color='k')
plt.xlabel('Semimajor axis length')
plt.ylabel('B4')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
plt.savefig('harmonics_'+out+label+'.png')

llim = np.nanmin(mag) - np.nanstd(mag)/2
hlim = np.nanmax(mag) + np.nanstd(mag)/2
# Plot surface brightness profile
plt.figure(figsize=(10, 6))
plt.ylim(llim,hlim)
plt.errorbar(isolist_wide.sma, mag,yerr=mag_err, fmt='o')
plt.title(gal_name+' '+label)
plt.xlabel('SMA [pix]')
plt.ylabel('$\mu$')
plt.gca().invert_yaxis()
plt.savefig('mag_'+out+label+'.png')

