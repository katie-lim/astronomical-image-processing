# %%
from uncertainties import unumpy
from load_data import *
from plot_data import *
from background_threshold import *
from sky_background import *
from source_detection import *
from photometry import *
from logNm import *

# %%

# Calculate the sky background
# and the threshold above which a pixel is considered a source
Nx, Ny = 8, 12
bg, threshold = calculateSkyBg(Nx, Ny)

# %%
# Load in the image
image = getImage()

image = image - bg # Subtract the background
image = doManualMasking(image) # Mask bad regions
image[np.ma.less(image, threshold)] = np.ma.masked # Mask regions below source threshold

# %%

# Run on a small section of the image
# xmin, xmax = 0, 1000
# xmin, xmax = 1000, 2000
# ymin, ymax = 0, 1000
xmin, xmax = 0, width
ymin, ymax = 0, height

image = image[ymin:ymax, xmin:xmax]


# %%
plotZScale(image.data)
plt.show()
#%%
plotMinMax(image.mask)
plt.show()
# %%

# Detect sources
sourceEllipses, apertureSums = detectSources(image)

# %%

plotZScale(originalImage[ymin:ymax, xmin:xmax])
plotEllipses(sourceEllipses)
plt.show()

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
