# %%
import numpy as np
from uncertainties import unumpy
from load_data import *
from plot_data import *
from background_threshold import *
from source_detection import *
from photometry import *



def gaussian2D(x, y, muX, muY, sigmaX, sigmaY, A):
    return A * np.exp(-((x - muX)**2 / (2. * sigmaX**2) + (y - muY)**2 / (2 * sigmaY**2)))



def generateGaussianSource(height, width, gaussianParams):
    x = np.arange(0, width)
    y = np.arange(0, height)

    # 2D variables instead of 1D
    x, y = np.meshgrid(x, y)

    z = gaussian2D(x, y, *gaussianParams)

    return z



def generateNoiseTestImage(height, width, bg=500, bgSigma=5):
    data = np.random.normal(bg, bgSigma, (height, width))

    return data



def generateGaussianTestImage(height, width, Nsources, bg=500, bgSigma=5, maxBrightness=5000):
    data = np.random.normal(bg, bgSigma, (height, width))

    for i in range(Nsources):
        x = int(width * np.random.random())
        y = int(height * np.random.random())


        sigma = 1/0.258 # 1" source
        sigmaX, sigmaY = sigma, sigma
        A = maxBrightness


        gaussian = generateGaussianSource(height, width, (x, y, sigmaX, sigmaY, A))

        data += gaussian


    return data


def detectSourcesInTestImage(data):
    threshold = getBackgroundThreshold(data)


    image = np.ma.masked_array(data, False, fill_value=0)
    image = maskBackground(image, threshold)


    plotMinMax(np.logical_not(image.mask))


    ellipses, apertureSums = detectSources(image)

    plotZScale(image.data)
    plotEllipses(ellipses)

    return ellipses, apertureSums




# %%

data = generateGaussianTestImage(1000, 1000, 80)
ellipses, apertureSums = detectSourcesInTestImage(data)


# Calculate magnitudes from the pixel sums
magnitudes = calcMagnitudes(apertureSums)

print(magnitudes)

# %%

data = generateNoiseTestImage(1000, 1000)
ellipses, apertureSums = detectSourcesInTestImage(data)

# %%

