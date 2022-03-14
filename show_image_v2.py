from get_data import *
from matplotlib.patches import Circle


'''
In order to eliminate the background, 
we have to find a reasonable value for the background

The upperlim that it takes is determined by eyes, and the actual 
threshold is determined using 5 standard deviation from the mean

Looking at the histogram of the background,
the distribution has a gaussian shape, therefore we fit a gaussian
'''
def gauss(data, A, x0, sigma):
    return A*np.exp(-(data - x0)**2 / (2 * sigma**2))

def hist_togetcap(data, upperlim, Nstd):

    flattendata = data.ravel()
    
    background = np.delete(flattendata, np.where(flattendata>upperlim))
    
    bin_heights, bin_borders = np.histogram(background, bins=1200, density=True)
    
    bin_widths = np.diff(bin_borders)
    
    bin_centers = bin_borders[:-1] + bin_widths / 2
    
    popt, pcov = curve_fit(gauss, bin_centers, bin_heights, p0=[1., 3400., 10.])

    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

    # plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
    # plt.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), label='fit', c='red')
    # plt.xlim(3300,3550)
    # plt.legend()
    # print('Fit Values: \nMean is', popt[1], '\nSigma is', popt[2])
    # print('Error is', np.sqrt(np.diag(pcov)))

    return popt[1] + popt[2] * Nstd
'''
The function returns the threshold of the background
'''
# hist_togetcap(cleandata, 3700, 5) 

def deletebackground(data, cap):
    data = np.ma.masked_array(data, data<cap, fill_value=0)
    return data

cap = hist_togetcap(cleandata, 3700, 5)

image = deletebackground(cleandata, cap)

def Plot(deepcleandata):

    interval = ZScaleInterval()
    
    vmin, vmax = interval.get_limits(deepcleandata)
    
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    
    fig = plt.figure(figsize=(3, 8), dpi=400)
   
    ax = fig.add_subplot(1, 1, 1)
   
    im = ax.imshow(deepcleandata, origin='lower', norm=norm, cmap='gray') 

# Plot(cleandata)



