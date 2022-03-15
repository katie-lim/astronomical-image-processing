# %%
from load_data import *
from plot_data import *
from background_threshold import *
from source_detection import *
from photometry import *

# Exclude the vertical line and artefacts (e.g. edge effects) from our analysis of the background
cleanImage = getCleanImage()

# Calculate the background threshold
threshold = getBackgroundThreshold(cleanImage).n

# Load in the image and remove bad sections so we can detect sources
image = getImage()
image = maskBackground(image, threshold)
image = maskVerticalLine(image, 1410, 1470)
image = cropImage(image, ymin=500, ymax=4500, xmax=2500)

# %%
# Detect sources
sourcePositions = findBrightestSources(image, 10)
# %%
plotCircles(image, sourcePositions)

# %%
# Photometry
apertureSums = [getApertureSum(image, *pos, 12, 24) for pos in sourcePositions]


# Some apertureSums are -ve
# due to the 12px aperture being too small
# Exclude these sources

sourcePositions = np.array(sourcePositions)
apertureSums = np.array(apertureSums)


sourcePositions = sourcePositions[apertureSums > 0]
apertureSums = apertureSums[apertureSums > 0]

# %%
# Calculate magnitudes from the pixel sums
magnitudes = convertToMagnitudes(apertureSums)

print("List of sources identified")
print("--------------------------------")
for ((x, y), mag) in zip(sourcePositions, magnitudes):
    print("x: %d, y: %d, magnitude: %.2f" % (x, y, mag))

# %%
plt.figure(dpi=400)
plt.hist(magnitudes, bins=30)
plt.xlabel("magnitude")
plt.ylabel("number of sources")
# %%
