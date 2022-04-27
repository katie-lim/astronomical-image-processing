# %%
import numpy as np
from uncertainties import unumpy
from load_data import *
from plot_data import *
from background_threshold import *
from sky_background import *
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



def generateNoiseTestImage(height, width, bgValue, bgSigma):
    data = np.random.normal(bgValue, bgSigma, (height, width))

    return data



def generateGaussianTestImage(height, width, Nsources, bgValue, bgSigma, sourceSigma, sourceAmplitude=5000):
    data = np.random.normal(bgValue, bgSigma, (height, width))

    for i in range(Nsources):
        x = int(width * np.random.random())
        y = int(height * np.random.random())

        sigma = sourceSigma
        sigmaX, sigmaY = sigma, sigma
        A = sourceAmplitude


        gaussian = generateGaussianSource(height, width, (x, y, sigmaX, sigmaY, A))

        data += gaussian


    return data


def detectSourcesInTestImage(data):

    # Initialise a masked_array containing the image
    # with no masked regions
    image = np.ma.masked_array(data, False)

    # Calculate the sky background
    bg, threshold = calculateSkyBg(image, 6, 6)

    image = image - bg # Subtract the background
    image[np.ma.less(image, threshold)] = np.ma.masked # Mask regions below source threshold


    print("Masked image")
    plotMinMax(np.logical_not(image.mask))
    plt.show()


    ellipses, fluxCounts = detectSources(image)


    print("Result")
    plotZScale(image.data)
    plotEllipses(ellipses)
    plt.show()

    return ellipses, fluxCounts


def calcExpectedFluxCount(sigmaX, sigmaY, A):
    return 2*np.pi*sigmaX*sigmaY*A



# %%

# Noise test image

data = generateNoiseTestImage(1000, 1000, 500, 20)
ellipses, fluxCounts = detectSourcesInTestImage(data)

# %%

# Gaussian test image

fwhm = 1/0.258 # 1" source
sigma = fwhm/2.355 # Convert FWHM to Gaussian sigma
A = 5000 # Source amplitude

data = generateGaussianTestImage(1000, 1000, 80, 500, 20, sigma, A)
ellipses, fluxCounts = detectSourcesInTestImage(data)


# Compute expected magnitude
expectedFluxCount = calcExpectedFluxCount(sigma, sigma, A)
expectedMagnitude = calcMagnitudes([expectedFluxCount])
print("Expected magnitude:", expectedMagnitude)


# Calculate magnitudes from the pixel sums
magnitudes = calcMagnitudes(fluxCounts)

print("Actual measured magnitudes:")
print(magnitudes)

# %%

# All 0s test image

data = np.zeros((1000, 1000))
ellipses, fluxCounts = detectSourcesInTestImage(data)

# %%
