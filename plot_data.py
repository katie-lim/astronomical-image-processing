import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from astropy.visualization import ImageNormalize, ZScaleInterval


def plotZScale(image, cmap="gray"):

    norm = ImageNormalize(image, ZScaleInterval())

    fig = plt.figure(figsize=(3, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(image, norm=norm, origin="lower", cmap=cmap)
    fig.colorbar(im, fraction=0.07, pad=0.04)


    height, width = image.shape
    plt.xlim(0, width)
    plt.ylim(0, height)



def plotMinMax(image, cmap="gray"):

    fig = plt.figure(figsize=(3, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(image, origin="lower", cmap=cmap)
    fig.colorbar(im, fraction=0.07, pad=0.04)


    height, width = image.shape
    plt.xlim(0, width)
    plt.ylim(0, height)



def plotCircles(image, sourcePositions):
    plotZScale(image.data)

    for (x, y) in sourcePositions:
        circ = Circle((x, y), 75, fc="#00000000", ec="blue")
        plt.gca().add_patch(circ)


def plotEllipse(ellipse):
    # Ellipse format given by opencv
    xy = ellipse[0]
    majorAxLength, minorAxLength = ellipse[1]
    angle = ellipse[2]


    ellipsePatch = Ellipse(xy, majorAxLength, minorAxLength, angle, fc="#00000000", ec="blue")
    plt.gca().add_patch(ellipsePatch)