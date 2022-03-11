import numpy as np
from load_data import *


def getPixelSumWithinRadius(imageWithMask, x, y, radius):
    image = imageWithMask.data
    height, width = image.shape

    counts = 0
    noOfPixels = 0

    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            if (dx**2 + dy**2) <= radius**2:
                newx = x + dx
                newy = y + dy

                if (0 <= newx < width) and (0 <= newy < height):
                    counts += image[newy, newx]
                    noOfPixels += 1

    return counts, noOfPixels



def getApertureSum(imageWithMask, x, y, radius, referenceApertureRadius):
    image = imageWithMask.data

    sourceCount, noSourcePixels = getPixelSumWithinRadius(image, x, y, radius)

    bgCount, noBgPixels = getPixelSumWithinRadius(image, x, y, referenceApertureRadius)
    bgCount -= sourceCount
    noBgPixels -= noSourcePixels


    sourceMinusBg = sourceCount - noSourcePixels * (bgCount/noBgPixels)


    return sourceMinusBg


def convertToMagnitudes(apertureSums):
    zeroPt = header["MAGZPT"]
    zeroPtErr = header["MAGZRR"]

    return zeroPt - 2.5*np.log10(apertureSums)
# %%
