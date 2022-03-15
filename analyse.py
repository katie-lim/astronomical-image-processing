# %%
from load_data import *
from plot_data import *
from background_threshold import *
from source_detection import *
from photometry import *
# %%
# Exclude the vertical line and artefacts (e.g. edge effects) from our analysis of the background
cleanImage = getCleanImage()

# Calculate the background threshold
threshold = getBackgroundThreshold(cleanImage).n
# %%
# Load in the image and remove bad sections so we can detect sources
image = getImage()
image = maskBackground(image, threshold)
image = maskVerticalLine(image, 1410, 1470)
image = cropImage(image, ymin=500, ymax=4500, xmax=2500)

# %%
# Detect sources
sourceEllipses = findBrightestSources(image, 10)
# %%

plotZScale(image.data)
plotEllipses(sourceEllipses)

# %%
# Photometry
apertureSums = getApertureSumsEllipses(image, sourceEllipses, 25)

# Calculate magnitudes from the pixel sums
magnitudes = convertToMagnitudes(apertureSums)


# %%
print("List of sources identified")
print("--------------------------------")

for (ellipse, mag) in zip(sourceEllipses, magnitudes):
    x, y = ellipse[0]
    print("x: %.2f, y: %.2f, magnitude: %.2f" % (x, y, mag))

# %%
plt.figure(dpi=400)
plt.hist(magnitudes, bins=30)
plt.xlabel("magnitude")
plt.ylabel("number of sources")
# %%
