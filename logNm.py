# %%
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy

def plotLogNm(magnitudes):
    mags = unumpy.nominal_values(magnitudes)

    mMin, mMax = np.min(mags), np.max(mags)
    m = np.linspace(mMin, mMax, 100)

    N = np.array([np.sum(mags <= mi) for mi in m])

    # Remove points where there was only 1 count
    # Because then N - sqrt(N) = 0, and log(0) is undefined
    m = m[N > 1]
    N = N[N > 1]



    Nerr = np.sqrt(N)
    logN = np.log10(N)
    logNupper = np.log10(N + Nerr)
    logNlower = np.log10(N - Nerr)



    plt.figure(dpi=400)
    plt.errorbar(m, logN, (logN - logNlower, logNupper - logN), fmt=".")
    plt.xlabel("m")
    plt.ylabel("log N(m)")
    plt.show()

# %%
