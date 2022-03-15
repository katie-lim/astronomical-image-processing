# %%
import numpy as np
from load_data import *
from ellipses import *


def getApertureSumEllipse(image, ellipse, delta):
    """
    Returns the sum of pixel counts within the given ellipse.

    Params
    -------
    image
    ellipse
    delta: The width of the annular reference aperture.
    """

    # Enlarge the ellipse to create an annular reference aperture
    widerEllipse = enlargeEllipse(ellipse, delta)


    # Get pixels within each ellipse
    sourcePixels = getEllipsePixels(image, ellipse)
    widerEllipsePixels = getEllipsePixels(image, widerEllipse)
    annulusPixels = np.logical_xor(sourcePixels, widerEllipsePixels)


    # Find the sum of counts
    data = image.data
    bgCnt = np.sum(data[annulusPixels])
    sourceCnt = np.sum(data[sourcePixels])

    Nbg = np.count_nonzero(annulusPixels)
    Nsource = np.count_nonzero(sourcePixels)


    # Subtract contribution from background
    sourceCnt -= Nsource * (bgCnt / Nbg)

    return sourceCnt


def getApertureSumsEllipses(image, ellipses, delta):
    return [getApertureSumEllipse(image, ellipse, delta) for ellipse in ellipses]


def convertToMagnitudes(apertureSums):
    zeroPt = header["MAGZPT"]
    zeroPtErr = header["MAGZRR"]

    return zeroPt - 2.5*np.log10(apertureSums)


# %%
