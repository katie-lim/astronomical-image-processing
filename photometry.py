# %%
import numpy as np
from load_data import *
from plot_data import *
from ellipses import *


def getApertureSumEllipse(image, ellipse, delta, plot=False):
    """
    Returns the sum of pixel counts within the given ellipse.

    Params
    -------
    image
    ellipse
    delta: The width of the annular reference aperture.
    plot: Whether to show a plot of the source and reference apertures.
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


    # Plot the result
    if plot:
        plotZScale(image.data, "gray")
        plotEllipses([ellipse, widerEllipse])
        x, y = ellipse[0]
        plt.title("source at (%.2f, %.2f)" % (x, y))

        # Zoom into this source
        boxSize = 100
        plt.xlim(x-boxSize, x+boxSize)
        plt.ylim(y-boxSize, y+boxSize)

        plt.show()


    return sourceCnt


def getApertureSumsEllipses(image, ellipses, delta):
    return [getApertureSumEllipse(image, ellipse, delta) for ellipse in ellipses]


def convertToMagnitudes(apertureSums):
    zeroPt = header["MAGZPT"]
    zeroPtErr = header["MAGZRR"]

    return zeroPt - 2.5*np.log10(apertureSums)


# %%
