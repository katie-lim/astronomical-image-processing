# %%
from numba import jit
import numpy as np
from uncertainties import unumpy

from load_data import *



def findBrightestSources(image, N):
    height, width = image.shape

    # Pixels in order of brightest to dimmest
    pixelIndices = np.flip(np.argsort(image.filled(0), axis=None))


    sourcePositions = []


    for index in pixelIndices:
        if len(sourcePositions) >= N:
            break

        y, x = np.unravel_index(index, image.shape)


        # Check whether this pixel is masked
        if (image[y, x] is np.ma.masked):
            continue


        # Store the x, y of this source
        sourcePositions.append((x, y))


        # Mask the circular region surrounding the pixel
        radius = 100

        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if (dx**2 + dy**2) <= radius**2:
                    newx = x + dx
                    newy = y + dy

                    if (0 <= newx < width) and (0 <= newy < height):
                        image[newy, newx] = np.ma.masked


    return sourcePositions



@jit(nopython=True)
def findBrightestSourcesFast(image, mask, N):
    height, width = image.shape

    # pixelIndices = np.flip(np.argsort(image.filled(0), axis=None))
    # numba doesn't support the filled function
    pixelIndices = np.flip(np.argsort(image.flatten()))


    sourcePositions = []


    for index in pixelIndices:
        if len(sourcePositions) >= N:
            break

        # y, x = np.unravel_index(index, image.shape)
        # numba doesn't support the unravel_index function
        # so do it manually
        y = index // width
        x = index % width


        # Check whether this pixel is masked
        if (mask[y, x] == True):
            continue


        # Store the x, y of this source
        sourcePositions.append((x, y))


        # Mask the circular region surrounding the pixel
        radius = 100

        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if (dx**2 + dy**2) <= radius**2:
                    newx = x + dx
                    newy = y + dy

                    if (0 <= newx < width) and (0 <= newy < height):
                        mask[newy, newx] = True


    return sourcePositions





# %%