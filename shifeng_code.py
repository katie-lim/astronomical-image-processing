from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, SqrtStretch,
                                   ImageNormalize)

filename = 'A1_mosaic.fits'


def find_max(index):
    image_data = fits.getdata(filename)
    image = image_data #500 was chosen by looking at the yscale
    index = unravel_index(np.argsort(image, axis = None)[-index],image.shape)
    # print('The value of the pixcel is', image[index[0],index[1]])
    # print(index)
    return index

def get_image(index):
    image_data = fits.getdata(filename)
    image = image_data[500::,::] #500 was chosen by looking at the yscale
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image)
    # Create an ImageNormalize object using a SqrtStretch object
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    # Display the image
    # d = image.ravel()
    # c = np.delete(d, np.where(d>=5000))
    # print(len(c))
    # plt.hist(c, bins=500 ,density=True, facecolor='g', alpha=0.75)    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(find_max(index)[1], find_max(index)[0]+500, s = 100, color='red')
    im = ax.imshow(image, origin='lower', norm=norm, cmap='gray') # cmap='gray_r' gives a darker image
    fig.colorbar(im)

get_image(1)