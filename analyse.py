# %%
from load_data import *
from background_threshold import *
from source_detection import *
from photometry import *


threshold = getBackgroundThreshold()

imageWithMask = getImageWithMask(originalImage, threshold)

imageWithMask = maskVerticalLine(imageWithMask, 1425, 1452, 3200, 3200)

image = cropImage(imageWithMask, ymin=500, ymax=4500, xmax=2500)

# %%
sourcePositions = findBrightestSourcesFast(image.data, image.mask, 200)

plotCircles(image, sourcePositions)

# %%
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
