# %%
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy
from scipy.optimize import curve_fit
from load_data import *



def getCleanImage():
    # Remove bad pixels
    # so they don't distort our analysis of the background
    image = getImage()
    image = maskVerticalLine(image, 1425, 1452)
    image = cropImage(image, ymin=500, ymax=4500, xmax=2500)


    # Remove masked pixels from our analysis
    pixels = image.data[image.mask == False]


    # Restrict to pixel values between 3300 & 3600
    pixels = pixels[(pixels > 3300) & (pixels < 3600)]


    return pixels



def getBackgroundThreshold(image, bins=100, Nsigma=5):
    """In order to eliminate the background,
we have to find a reasonable value for the background

Looking at the histogram of the background,
the distribution has a gaussian shape, therefore we fit a gaussian

The threshold is determined using 5 standard deviation from the mean

The function returns the threshold of the background"""


    image = image.flatten()


    plt.figure(dpi=400)
    heights, bins, patches = plt.hist(image, bins, label="data")
    binCentres = (bins[1:] + bins[:-1])/2


    # Fit a Gaussian
    def gaussian(x, mu, sigma, A):
        return A*np.exp(-(x - mu)**2 / (2 * sigma**2))


    p0 = [np.mean(image), np.std(image), np.max(heights)]
    params, cov = curve_fit(gaussian, binCentres, heights, p0)
    errors = np.sqrt(np.diag(cov))
    paramsWithErrors = unumpy.uarray(params, errors)
    mu, sigma, A = paramsWithErrors

    threshold = mu + Nsigma*sigma


    # Raise an error if we were unable to fit a Gaussian
    if (np.any(np.isinf(errors))):
        raise Exception("Unable to fit a Gaussian to the background.")



    # Plot the resulting fit
    x = np.linspace(binCentres[0], binCentres[-1], 500)
    plt.plot(x, gaussian(x, *params), label="Gaussian fit")
    plt.vlines(threshold.n, 0, heights.max(), color="r", linestyles="dotted", label="5$\sigma$ threshold")


    plt.xlabel("pixel value")
    ylabel = "number of pixels"
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("histogram of pixel values")
    plt.show()


    # Print the results
    print("Gaussian fit parameters mu, sigma, A:")
    print([p.format("%.2u") for p in paramsWithErrors])
    print("")


    return threshold
# %%