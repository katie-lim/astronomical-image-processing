# %%
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from uncertainties import unumpy
from astropy.visualization import ImageNormalize, ZScaleInterval
from matplotlib.patches import Circle



def plotZScale(image):
    height, width = image.shape

    norm = ImageNormalize(image, ZScaleInterval())

    fig = plt.figure(figsize=(3, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image, norm=norm, origin="lower", cmap="gray")
    plt.xlim(0, width)
    plt.ylim(0, height)

    fig.colorbar(im)



def getImageWithMask(image, threshold):
    return np.ma.masked_array(image, image < threshold, fill_value=0)



def maskVerticalLine(imageWithMask, xmin, xmax, ylower, yupper):
    image = imageWithMask.data
    height, width = image.shape

    for y in range(0, height):
        if (ylower <= y <= yupper):
            continue

        for x in range (xmin, xmax):
            imageWithMask[y, x] = np.ma.masked

    return imageWithMask




def cropImage(image, xmin=0, xmax=None, ymin=0, ymax=None):
    return image[ymin:ymax, xmin:xmax]



def findBrightestSources(imageWithMask, N):
    image = imageWithMask
    height, width = image.shape

    # Highest to lowest
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

    # Highest to lowest
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




def plotCircles(imageWithMask, sourcePositions):
    plotZScale(imageWithMask.data)

    for (x, y) in sourcePositions:
        circ = Circle((x, y), 75, fc="#00000000", ec="blue")
        plt.gca().add_patch(circ)


# %%