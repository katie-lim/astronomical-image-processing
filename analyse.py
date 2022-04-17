# %%
from uncertainties import unumpy
from load_data import *
from plot_data import *
from background_threshold import *
from source_detection import *
from photometry import *
from logNm import *

# %%
# Exclude the vertical line and artefacts (e.g. edge effects) from our analysis of the background
cleanPixels = getCleanPixels()

# Calculate the background threshold
threshold = getBackgroundThreshold(cleanPixels, Nbins=25).n
# %%
# Load in the image and remove bad sections so we can detect sources
image = getImage()
image = doManualMasking(image)

image[image < threshold] = np.ma.masked
# %%

# Run on a small section of the image

image = image[0:1000, 0:1000]
# image = image[1000:2000, 0000:1000]
# image = image[0:2000, 0000:1000]


# %%
maskedImage = image.filled(0)
plotZScale(maskedImage)
# %%

# Detect sources
sourceEllipses, apertureSums = detectSources(image)

# %%

plotZScale(image.data)
plotEllipses(sourceEllipses)

# %%

plotZScale(maskedImage)
plotEllipses(sourceEllipses)


# %%
# Photometry
# Add errors to the counts
aperSumsWithErr = unumpy.uarray(apertureSums, np.sqrt(apertureSums))

# Calculate magnitudes from the pixel sums
magnitudes = calcMagnitudes(apertureSums)


# %%
print("List of sources identified")
print("--------------------------------")

for (ellipse, mag) in zip(sourceEllipses, magnitudes):
    x, y = ellipse[0]
    print("x: %.2f, y: %.2f, magnitude: %s" % (x, y, mag.format("%.2u")))

# %%

gradient, yintercept = fitLogNm(magnitudes, mCutoff=17)

# %%

# sourceEllipses, aperSumsWithErr, magnitudes = loadCatalogue()
saveCatalogue(sourceEllipses, aperSumsWithErr, magnitudes)

# %%
