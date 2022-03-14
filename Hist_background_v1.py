from astropy.io import fits
import numpy as np
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, SqrtStretch,
                                   ImageNormalize)


image = fits.getdata("A1_mosaic.fits")
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(image)
# Create an ImageNormalize object using a SqrtStretch object
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
# Display the image

a = image[::,::]
b = a.ravel()
p = np.delete(b, np.where(b>6000))

plt.hist(p, bins=1500 ,density=True, facecolor='g', alpha=0.75)    
plt.xlim(3000,4000)
