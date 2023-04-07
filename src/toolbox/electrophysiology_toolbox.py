from scipy.signal import butter, lfilter
import numpy as np
from typing import Tuple
import logging

logger=logging.getLogger()


def extract_lfp(raw: np.array, fs: float, 
    filter_freq: float=200, out_fs: float = 500, order: int = 3
) -> Tuple[np.array, float]:
    """
    Extracts the lfp from the @raw electrophysiology signal sampled at frequency @fs
    by using a lower butter low pass of order @order with max frequency @filter_freq.
    The final frequency is resampled at around min(@fs, @max_out_fs).

    Returns the lfp and the sampled frequency of the lfp
    """

    filtera_lfp, filterb_lfp = butter(order, filter_freq, fs=fs, btype='low', analog=False)
    lfp = lfilter(filtera_lfp, filterb_lfp, raw)

    if(fs > out_fs):
        return lfp[::int(fs/out_fs)], fs/int(fs/out_fs)
    else:
        return lfp, fs  

def extract_mu(
        raw: np.array, fs: float, 
        filter_low_freq=300, filter_high_freq=6000, filter_refreq=1000, 
        out_fs: float = 500, order=3,
)  -> Tuple[np.array, float]:
    """
    Extracts the multi-unit from the @raw electrophysiology signal sampled at frequency @fs
    by:
     1. using a butter band pass of order @order with band [@filter_low_freq, @filter_high_freq].
     2. taking the absolute value
     3. using a butter low pass of order @order with max frequency @filter_refreq
    The final frequency is resampled at around min(@fs, @max_out_fs).

    Returns the mu and the sampled frequency of the mu
    """

    filtera_mu_tmp, filterb_mu_tmp = butter(order, [filter_low_freq, filter_high_freq], fs=fs, btype='band', analog=False)
    mu_tmp = lfilter(filtera_mu_tmp, filterb_mu_tmp, raw)

    mu_abs=np.abs(mu_tmp)

    filtera_mu, filterb_mu = butter(order, filter_refreq, fs=fs, btype='low', analog=False)
    mu = lfilter(filtera_mu, filterb_mu, mu_abs)
    
    if(fs > out_fs):
        return mu[::int(fs/out_fs)], fs/int(fs/out_fs)
    else:
        return mu, fs