import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from astropy.visualization import ImageNormalize, ZScaleInterval


# Styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams.update({'font.size': 12})



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



def plotZoomedIn(image, x, y, boxWidth, boxHeight, zscale=True, show=True):
    if zscale:
        plotZScale(image)
    else:
        plotMinMax(image)

    plt.xlim(x - boxWidth//2, x + boxWidth//2)
    plt.ylim(y - boxHeight//2, y + boxHeight//2)

    if show:
        plt.show()



def plotCircles(image, sourcePositions):
    plotZScale(image.data)

    for (x, y) in sourcePositions:
        circ = Circle((x, y), 75, fc="#00000000", ec="blue")
        plt.gca().add_patch(circ)


def plotEllipse(ellipse):
    # Ellipse format given by opencv
    (x, y), (majorAxLength, minorAxLength), angle = ellipse


    ellipsePatch = Ellipse((x, y), majorAxLength, minorAxLength, angle, fc="#00000000", ec="blue")
    plt.gca().add_patch(ellipsePatch)


def plotEllipses(ellipses):
    for ellipse in ellipses:
        plotEllipse(ellipse)