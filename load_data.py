import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval


filePath = "A1_mosaic.fits"


def loadImage(filePath):
    return fits.getdata(filePath)


def loadHeader(filePath):
    return fits.getheader(filePath, 0)


originalImage = loadImage(filePath)
header = loadHeader(filePath)



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



def plotZScale(image):
    height, width = image.shape

    norm = ImageNormalize(image, ZScaleInterval())

    fig = plt.figure(figsize=(3, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image, norm=norm, origin="lower", cmap="gray")
    plt.xlim(0, width)
    plt.ylim(0, height)

    fig.colorbar(im)
