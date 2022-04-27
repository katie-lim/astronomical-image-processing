import numpy as np
import pandas as pd
from uncertainties import unumpy
from astropy.io import fits


filePath = "A1_mosaic.fits"


def loadData(filePath):
    return fits.getdata(filePath)


def loadHeader(filePath):
    return fits.getheader(filePath, 0)


originalImage = loadData(filePath)
header = loadHeader(filePath)
height, width = originalImage.shape



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



def maskCircle(image, x, y, radius):

    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if (dx**2 + dy**2) < radius**2:
                image[y + dy, x + dx] = np.ma.masked

    return image



def maskRectangle(image, xmin, xmax, ymin, ymax):

    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            image[y, x] = np.ma.masked

    return image



def maskEdges(image, xmin=0, xmax=None, ymin=0, ymax=None):
    # image[ymin:ymax, xmin:xmax] = np.ma.masked
    height, width = image.shape

    image[0:ymin, : ] = np.ma.masked
    image[ymax:height, : ] = np.ma.masked
    image[ : ,0:xmin] = np.ma.masked
    image[ : ,xmax:width] = np.ma.masked

    return image



def cropImage(image, xmin=0, xmax=None, ymin=0, ymax=None):
    return image[ymin:ymax, xmin:xmax]



def doManualMasking(image):
    image = maskVerticalLine(image, 1410, 1470)
    image = maskCircle(image, 1420, 3220, 400)
    image = maskEdges(image, ymin=200, ymax=4400, xmin=200, xmax=2400)

    bleedingSources = [(2091, 2185, 3704, 3809),
                        (717, 867, 3195, 3423),
                        (917, 1034, 2695, 2843),
                        (851, 968, 2217, 2364),
                        (1000, 1670, 0, 537)]

    for (xmin, xmax, ymin, ymax) in bleedingSources:
        image = maskRectangle(image, xmin, xmax, ymin, ymax)

    return image




def saveCatalogue(ellipses, fluxCounts, magnitudes):
    # Ellipses
    xy, axLengths, angles = list(zip(*ellipses))
    x, y = list(zip(*xy))
    majorAxLengths, minorAxLengths = list(zip(*axLengths))

    # Photometry
    fluxCountsErrs = np.sqrt(fluxCounts)

    mags = unumpy.nominal_values(magnitudes)
    magsErrs = unumpy.std_devs(magnitudes)


    # Save results in a csv
    df = pd.DataFrame({"x": x, "y": y, "majoraxislength": majorAxLengths, "minoraxislength": minorAxLengths, "angle": angles, "count": fluxCounts, "counterr": fluxCountsErrs, "magnitude": mags, "magnitudeerr": magsErrs})

    df.to_csv("catalogue.csv")



def loadCatalogue():
    df = pd.read_csv("catalogue.csv", index_col=0)


    x, y, majorAxLengths, minorAxLengths, angles, fluxCounts, fluxCountsErrs, mags, magsErrs = df.to_numpy().T

    ellipses = []

    for i in range(len(x)):
        ellipse = ((x[i], y[i]), (majorAxLengths[i], minorAxLengths[i]), angles[i])
        ellipses.append(ellipse)

    fluxCountsWithErrs = unumpy.uarray(fluxCounts, fluxCountsErrs)
    magnitudes = unumpy.uarray(mags, magsErrs)


    return ellipses, fluxCountsWithErrs, magnitudes


# %%
