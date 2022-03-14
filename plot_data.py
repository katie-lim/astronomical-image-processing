import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import ImageNormalize, ZScaleInterval


def plotZScale(image):

    norm = ImageNormalize(image, ZScaleInterval())

    fig = plt.figure(figsize=(3, 8), dpi=400)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(image, norm=norm, origin="lower", cmap="gray")
    fig.colorbar(im)


    height, width = image.shape
    plt.xlim(0, width)
    plt.ylim(0, height)



def plotCircles(image, sourcePositions):
    plotZScale(image.data)

    for (x, y) in sourcePositions:
        circ = Circle((x, y), 75, fc="#00000000", ec="blue")
        plt.gca().add_patch(circ)
