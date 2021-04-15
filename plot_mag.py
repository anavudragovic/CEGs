#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:06:40 2021

@author: anavudragovic

Plots all radial profiles of NGC1270 galaxy measured by ellipsefitting.py in one plot
Errors are omitted since they are very large and make too much noise in the plot
"""

import sys
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import re

path = "/Users/anavudragovic/vidojevica/compact_ellipticals_at_Milankovic"
os.chdir(path)
mags='mag_iclsub*_ngc12710-2.txt'
bkgs='mag_ngc12710-2_b*.txt'
nfiles = len(glob.glob(mags)+glob.glob(bkgs))

names = []
numlines = []
i=0
for fn in glob.glob(mags)+glob.glob(bkgs):
    with open(fn) as f:
        names=np.append(names, fn)
        #names[i]=[sum(1 for line in f if line.strip() and not line.startswith('#'))  ]  
        numlines = np.append(numlines, sum(1 for line in f if line.strip() and not line.startswith('#')))
        i+=1
 
rows=len(names)
cols = int(np.max(numlines))
sma = [[0. for i in range(cols)] for j in range(rows)]
mag = [[0. for i in range(cols)] for j in range(rows)]
mag_err = [[0. for i in range(cols)] for j in range(rows)]
i=0
for imfile, file in enumerate(glob.glob(mags)+glob.glob(bkgs)):
    print(file)
    ellip = np.genfromtxt(file, names="sma,mag,mag_err",usecols=(0,-2,-1),delimiter=" ",skip_header=1)
    sma[i] = ellip['sma']
    mag[i] = ellip['mag']
    mag_err[i] = ellip['mag_err']
    i+=1



fig = plt.figure(figsize=(10,7.5))
#viridis = cm.get_cmap('viridis', 256)
#colors = viridis(np.linspace(0, 1, 12))
colors=cm.rainbow(np.linspace(0, 1, len(mag)))
llim = -7.2 #-12
hlim = -2.

i=0
for x, y, yerr, c in zip(sma, mag, mag_err, colors):
    #print(len(x),len(y))
    plt.ylim(llim,hlim)
    plt.xlim(40,150)
    plt.plot(x, y,color=c,alpha=0.9,label=names[i].replace('_ngc12710-2','').replace('mag_','').replace('.txt',''),ls='-' if (i%2==0) else '--')
    plt.xlabel('sma [pix]')
    plt.ylabel('$\mu$')
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')
    i+=1
plt.savefig('RP_zoom.png')    
