from astropy.io import fits

filePath = "A1_mosaic.fits"


def loadImage(filePath):
    return fits.getdata(filePath)


def loadHeader(filePath):
    return fits.getheader(filePath, 0)


originalImage = loadImage(filePath)
header = loadHeader(filePath)