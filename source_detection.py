# %%
import cv2
from skimage.segmentation import flood
import numpy as np
from numba import jit
from uncertainties import unumpy
from plot_data import *
from load_data import *
from ellipses import *


def findBrightestSources(image, N):

    # Pixels in order of brightest to dimmest
    pixelIndices = np.flip(np.argsort(image.filled(0), axis=None))


    sourceEllipses = []


    for index in pixelIndices:
        if len(sourceEllipses) >= N:
            break

        y, x = np.unravel_index(index, image.shape)


        # Check whether this pixel is masked
        if (image[y, x] is np.ma.masked):
            continue


        # Fit an ellipse to the source and store it
        ellipse = fitEllipseToSource(image, x, y)
        sourceEllipses.append(ellipse)


        # Mask the region contained within the ellipse
        ellipsePixels = getEllipsePixels(image, ellipse)
        print(ellipsePixels)



        image.mask = np.logical_or(image.mask, ellipsePixels)


    return sourceEllipses




def fitEllipseToSource(image, x, y):
    cleanData = getPixelsWithinSource(image, x, y)
    ellipse = fitEllipseToCleanData(cleanData, x, y)

    largerEllipse = enlargeEllipse(ellipse, 25)


    plotZScale(image.data, "gray")
    plotEllipse(ellipse)
    plotEllipse(largerEllipse)
    plt.title("source at (%d, %d)" % (x, y))

    # Zoom into this source
    boxSize = 100
    plt.xlim(x-boxSize, x+boxSize)
    plt.ylim(y-boxSize, y+boxSize)

    plt.show()


    return ellipse



def getPixelsWithinSource(image, x, y):

    # Turn the image into a series of 0s and 1s
    # based on whether pixels are source or bg/masked
    mask = image.mask
    cleanData = np.logical_not(mask).astype(np.uint8)



    # Find the group of pixels connected to the source using skimage.segmentation.flood

    # Consider pixels to be connected if they're connected vertically or horizontally
    footprint = [[0,1,0],
                [1,1,1],
                [0,1,0]]


    # Beware: position is (y, x), not (x, y)!
    cleanData = flood(cleanData, (y, x), footprint=footprint).astype(np.uint8)


    return cleanData


def fitEllipseToCleanData(cleanData, x, y):
    # Find contours
    contours, hierarchy = cv2.findContours(cleanData, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

    # cv2.RETR_EXTERNAL to find the outermost contour


    # plt.figure(dpi=400)
    # plt.imshow(cv2.drawContours(np.zeros(data.shape), contours, -1, (255, 255, 255), 2))
    # plt.show()


    if (len(contours) > 1):
        raise Exception("More than one contour was found for source at (%d, %d)." % (x, y))


    # Fit an ellipse
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)


    return ellipse

# %%