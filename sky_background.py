# %%
from load_data import *
from plot_data import *
from background_threshold import *

from scipy import interpolate

# %%
# Exclude the vertical line and artefacts (e.g. edge effects) from our analysis of the background
cleanPixels = getCleanPixels()

# Calculate background threshold for the entire image
threshold = getBackgroundThreshold(cleanPixels, Nsigma=5).n

# %%

# Load in the image
image = getImage()
image = doManualMasking(image)
image[image > threshold] = np.ma.masked # Mask bright pixels (sources) so they don't affect our calculation of background

# %%

def calculateSkyBg(image, Nx, Ny):
    height, width = image.shape
    xSize, ySize = width // Nx, height // Ny

    x, y = xSize, ySize


    blockX = np.arange(xSize//2, width, xSize)
    blockY = np.arange(ySize//2, height, ySize)
    blockBg = []


    # Plot the blocks
    plotZScale(image)
    plt.vlines(blockX, 0, height)
    plt.hlines(blockY, 0, width)
    plt.show()


    # Calculate the background value in each block
    while (y < height):
        x = xSize

        while (x < width):
            block = image[y - ySize//2:y + ySize//2, x - xSize//2:x + xSize//2]

            median = np.ma.median(block)
            std = np.ma.std(block)

            if (median):
                blockBg.append(median)
            else:
                # If the block is entirely masked, we can't calculate the median
                # so just use the previous block's median
                # (this doesn't affect our results because masked pixels are ignored when detecting sources)
                blockBg.append(blockBg[-1])


            # Debugging
            # print((x, y))
            # plotZoomedIn(image, x, y, xSize, ySize, zscale=True)
            # print("Mean: %.2f, Median: %.2f, Std: %.2f" % (mean, median, std))


            x += xSize

        y += ySize


    blockBg = np.array(blockBg).reshape((Ny, Nx))

    # Interpolate between the blocks to calculate the sky background
    f = interpolate.RectBivariateSpline(blockX, blockY, blockBg.T)

    imageX = np.arange(width)
    imageY = np.arange(height)

    bg = f(imageX, imageY).T

    print(blockX)
    print(blockY)


    return bg

# %%
Nx, Ny = 8, 12

bg = calculateSkyBg(image, Nx, Ny)

# %%

# Plot the background
plotZScale(bg)
plt.show()

# %%

# Plot the background, showing masked regions
bgMasked = np.ma.MaskedArray(bg, False)
bgMasked = doManualMasking(bgMasked)

plotZScale(bgMasked)
plt.show()

# %%
plotZScale(image.data)
# %%
