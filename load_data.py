import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval


filePath = "A1_mosaic.fits"


def loadData(filePath):
    return fits.getdata(filePath)


def loadHeader(filePath):
    return fits.getheader(filePath, 0)


originalImage = loadData(filePath)
header = loadHeader(filePath)



def getImage():
    return np.ma.masked_array(originalImage, False, fill_value=0)



def maskBackground(image, threshold):
    data = image.data

    return np.ma.masked_array(data, data < threshold, fill_value=0)



def maskVerticalLine(image, xmin, xmax):
    height, width = image.shape

    for y in range(0, height):
        for x in range (xmin, xmax+1):
            image[y, x] = np.ma.masked

    return image



def cropImage(image, xmin=0, xmax=None, ymin=0, ymax=None):
    return image[ymin:ymax, xmin:xmax]
