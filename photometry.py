# %%
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy
from load_data import *
from plot_data import *
from ellipses import *


def getfluxCount(image, sourcePixels, ellipse, plot=False):

    # Find the sum of counts
    sourceCnt = np.sum(image.data[sourcePixels])

    # Plot the result
    if plot:
        x, y = ellipse[0]

        plotZoomedIn(image.data, x, y, 200, 200, show=False)
        plotEllipses(ellipse)

        plt.title("source at (%.2f, %.2f)" % (x, y))
        plt.show()


    return sourceCnt


def calcMagnitudes(fluxCounts):

    # Add errors to the flux counts
    fluxCountsWithErr = unumpy.uarray(fluxCounts, np.sqrt(fluxCounts))

    zeroPt = ufloat(header["MAGZPT"], header["MAGZRR"])
    magnitudes = zeroPt - 2.5*unumpy.log10(fluxCountsWithErr)

    return magnitudes

# %%
