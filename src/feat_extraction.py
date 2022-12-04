import numpy as np

from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps

# Multitaper method
def bandpower_multitaper(data, sf, method, band, relative=False):
    band = np.asarray(band)
    low, high = band
    
    if method == 'multitaper':
        psd_trial, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                                normalization='full', verbose=0)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd_trial[idx_band], dx=freq_res)
        
    return bp
