# %%
from load_data import *
from background_threshold import *
from source_detection import *

threshold = getBackgroundThreshold()

imageWithMask = getImageWithMask(originalImage, threshold)

imageWithMask = maskVerticalLine(imageWithMask, 1425, 1452, 3200, 3200)

image = cropImage(imageWithMask, ymin=500, ymax=4500, xmax=2500)

# %%
sourcePositions = findBrightestSourcesFast(image.data, image.mask, 200)

plotCircles(image, sourcePositions)

# %%
