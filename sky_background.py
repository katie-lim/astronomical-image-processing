# %%
from load_data import *
from plot_data import *
from background_threshold import *

from scipy import interpolate

# %%

def calculateSkyBg(Nx, Ny, Nsigma=5):
    # Load in the image
    image = getImage()
    image = doManualMasking(image)


    height, width = image.shape
    xSize, ySize = width // Nx, height // Ny

    x, y = xSize, ySize


    blockX = np.arange(xSize//2, width, xSize)
    blockY = np.arange(ySize//2, height, ySize)
    blockBg = []
    blockSigma = []


    # Plot the blocks
    print("Calculating sky background.")
    plotZScale(image)
    plt.vlines(blockX, 0, height, color="b", linestyles="dashed", lw=1)
    plt.hlines(blockY, 0, width, color="b", linestyles="dashed", lw=1)
    # plt.savefig("plots/sky_background_blocks.svg")
    plt.show()


    # Calculate the background value in each block
    while (y < height):
        x = xSize

        while (x < width):
            block = image[y - ySize//2:y + ySize//2, x - xSize//2:x + xSize//2]


            # If the block isn't completely masked
            if (not np.all(block.mask)):

                # Fit a Gaussian to the pixels in the block
                validPixels = block[block.mask == False]

                mu, sigma, A = getBackgroundThreshold(validPixels.flatten(), Nbins=25, returnFit=True, plot=False)

                blockBg.append(mu)
                blockSigma.append(sigma)

            else:
                # If the block is entirely masked, we can't calculate the bg
                # so just use the previous block's values
                # (this doesn't affect our results because masked pixels are ignored when detecting sources)
                blockBg.append(blockBg[-1])
                blockSigma.append(blockSigma[-1])


            # Debugging
            # print((x, y))
            # if std: plt.hist(block[block.mask == False], bins=30)
            # plotZoomedIn(image, x, y, xSize, ySize, zscale=True)
            # print("Mean: %.2f, Median: %.2f, Std: %.2f" % (mean, median, std))


            x += xSize

        y += ySize


    blockBg = np.array(blockBg).reshape((Ny, Nx))
    blockSigma = np.array(blockSigma).reshape((Ny, Nx))

    # Interpolate between the blocks to calculate the sky background
    f = interpolate.RectBivariateSpline(blockX, blockY, blockBg.T)

    imageX = np.arange(width)
    imageY = np.arange(height)

    bg = f(imageX, imageY).T


    # Calculate the threshold above which a pixel is considered a source
    # Take the threshold to be 5 sigma (by default)
    blockThreshold = blockSigma * Nsigma

    # Once again, interpolate between the blocks
    f = interpolate.RectBivariateSpline(blockX, blockY, blockThreshold.T)

    threshold = f(imageX, imageY).T


    return bg, threshold

# %%

# Nx, Ny = 8, 12

# bg, threshold = calculateSkyBg(Nx, Ny)

# # %%

# # Plot the background
# plotZScale(bg)
# plt.show()

# # %%

# # Plot the background, showing masked regions
# bgMasked = np.ma.MaskedArray(bg, False)
# bgMasked = doManualMasking(bgMasked)

# plotZScale(bgMasked)
# plt.show()

# # %%
# # Plot the source threshold (5 sigma)
# plotZScale(threshold)
# plt.show()
# # %%

# # Plot the source threshold, showing masked regions
# thresholdMasked = np.ma.MaskedArray(threshold, False)
# thresholdMasked = doManualMasking(thresholdMasked)

# plotZScale(thresholdMasked)
# plt.show()
# # %%
