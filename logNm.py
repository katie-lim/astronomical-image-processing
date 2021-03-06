# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from uncertainties import unumpy
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
from photutils.detection import DAOStarFinder




def getLogNmValues(magnitudes):
    mags = unumpy.nominal_values(magnitudes)

    mMin, mMax = np.min(mags), np.max(mags)
    m = np.linspace(mMin, mMax, 100)

    N = np.array([np.sum(mags <= mi) for mi in m])

    # Remove points where there was only 1 count
    # Because then N - sqrt(N) = 0, and log(0) is undefined
    m = m[N > 1]
    N = N[N > 1]


    NwithErr = unumpy.uarray(N, np.sqrt(N))
    logN = unumpy.log10(NwithErr)


    return m, logN




def fitLogNm(magnitudes, mCutoff):
    m, logN = getLogNmValues(magnitudes)


    # Remove points above the specified magnitude threshold
    # to account for incompleteness
    if not (mCutoff is None):
        logN = logN[m < mCutoff]
        m = m[m < mCutoff]


    logNvals = unumpy.nominal_values(logN)
    logNerrs = unumpy.std_devs(logN)


    # Fit a straight line
    p0 = [0.33, -2.5]
    params, cov = np.polyfit(m, logNvals, 1, w=1/logNerrs, cov=True)
    errors = np.sqrt(np.diag(cov))
    paramsWithErrors = unumpy.uarray(params, errors)


    gradient, yintercept = paramsWithErrors
    print("log N(m) = %.4fm %.4f" % (params[0], params[1]))
    print("Gradient = ", gradient)
    print("Y-intercept = ", yintercept)


    plotLogNm(magnitudes)
    # plt.plot(m, f(m, *p0), label="initial guess")
    plt.plot(m, np.poly1d(params)(m), label="fit", zorder=100, color="k")
    plt.vlines(mCutoff, -0.2, 3.1, color="#eb0c00", linestyles="dashed", lw=1)
    plt.legend()
    plt.savefig("plots/logNm.svg")
    plt.show()




def plotLogNm(magnitudes):
    m, logN = getLogNmValues(magnitudes)


    logNvals = unumpy.nominal_values(logN)
    logNerrs = unumpy.std_devs(logN)


    plt.figure(dpi=400)
    plt.errorbar(m, logNvals, logNerrs, fmt=".", label="data", color="#4b4bfe", capsize=2)
    plt.xlabel("m")
    plt.ylabel("log N(m)")
    # plt.show()








def getLogNmValuesAsym(magnitudes, mCutoff=None):
    mags = unumpy.nominal_values(magnitudes)

    mMin, mMax = np.min(mags), np.max(mags)
    m = np.linspace(mMin, mMax, 100)

    N = np.array([np.sum(mags <= mi) for mi in m])

    # Remove points where there was only 1 count
    # Because then N - sqrt(N) = 0, and log(0) is undefined
    m = m[N > 1]
    N = N[N > 1]


    # Remove points above the specified magnitude threshold
    # to account for incompleteness
    if not (mCutoff is None):
        N = N[m < mCutoff]
        m = m[m < mCutoff]


    Nerr = np.sqrt(N)
    logN = np.log10(N)
    logNupper = np.log10(N + Nerr)
    logNlower = np.log10(N - Nerr)

    sigmaLow = logN - logNlower
    sigmaUpp = logNupper - logN


    return m, logN, sigmaLow, sigmaUpp



def plotLogNmAsym(magnitudes):
    m, logN, sigmaLow, sigmaUpp = getLogNmValuesAsym(magnitudes)


    plt.figure(dpi=400)
    plt.errorbar(m, logN, (sigmaLow, sigmaUpp), fmt=".", label="data", color="#4b4bfe", capsize=2)
    plt.xlabel("m")
    plt.ylabel("log N(m)")
    # plt.show()



def fitLogNmAsym(magnitudes, mCutoff):
    m, logN, sigmaLow, sigmaUpp = getLogNmValuesAsym(magnitudes, mCutoff)


    # A straight line
    def f(x, grad, c):
        return grad*x + c


    # https://stackoverflow.com/questions/19116519/scipy-optimize-curvefit-asymmetric-error-in-fit

    # The function to minimise: sums of squares,
    # weighted by the error on each data point, sigma
    def loss_function(params):
        error = (logN - f(m, *params))
        error_neg = (error < 0)

        # Use sigmaUpp when the error is positive
        # and use sigmaLow when the error is negative

        error_squared = error**2 / (error_neg * sigmaLow + (1 - error_neg) * sigmaUpp)

        return error_squared.sum()


    x0 = [0.3, -3]
    x = fmin(loss_function, x0)

    print("log N(m) = %.4fm %.4f" % (x[0], x[1]))


    plotLogNmAsym(magnitudes)
    # plt.plot(m, f(m, *x0), label="initial guess")
    plt.plot(m, f(m, *x), label="fit", zorder=100, color="k")
    plt.vlines(mCutoff, -0.2, 3.1, color="#eb0c00", linestyles="dashed", lw=1)
    plt.legend()
    plt.savefig("plots/logNm.svg")
    plt.show()




def daoFind(cleanimage, Nsigma):
    #output the number count plot using Photutils
    data = cleanimage.data

    mean, median, std = sigma_clipped_stats(data)

    daofind = DAOStarFinder(fwhm=1.0, threshold = Nsigma * std)

    sources = daofind(data - Nsigma * std)

    for col in sources.colnames:

        sources[col].info.format = '%.8g'

    # print(sources)

    magn = sources['mag']

    gradient, yintercept = fitLogNmAsym(magn, mCutoff=None)


# %%
