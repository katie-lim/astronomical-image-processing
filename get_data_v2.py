from astropy.io import fits
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from numpy import unravel_index
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, SqrtStretch, ImageNormalize)

file = 'A1_mosaic.fits'

def get_data(filename):
    return fits.getdata(filename)

data = get_data(file) # this is the total dataset that we need to filter out a few points




def cut_data(data, xmin, xmax, ymin, ymax):
    return data[ymin:ymax, xmin:xmax]

cutdata = ma.array(cut_data(data, 0, 2570, 500, 4611)) # this is the data without the first 500 yaxis and put in a ma array in order to use mask function




def clean_data(cutdata, xlower, xupper):
    y, x = cutdata.shape

    for i in range (0, y-1):

        for k in range (0, x-1):

            if  k in range(xlower, xupper):
                cutdata[i, k] = ma.masked
                k+=1
            else:
                k+=1

    cleandata = cutdata
    return cleandata

cleandata = clean_data(cutdata, 1425, 1455) # this is the dataset that excludes the vertical line and the first 500 yaxis

