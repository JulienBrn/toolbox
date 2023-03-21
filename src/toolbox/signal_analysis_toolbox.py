import numpy as np
import pandas as pd
import logging
import tqdm

logger= logging.getLogger(__name__)

def remove_artefacts(signal, factor=2) -> None:
    """
    Removes artefacts from signal by removing recursively all values signal[i] such that 
    abs(signal[i-1] * factor) < signal[i] and abs(signal[i+1] * factor) < signal[i]
    """
    for i in range(1, len(signal)-1):
        if abs(signal[i-1] * factor) < abs(signal[i]) and abs(signal[i+1] * factor) < abs(signal[i]):
            signal[i]=(signal[i-1] + signal[i+1])/2
            i=i-1

def replace_artefacts_with_nans(sig, fs, deviation_factor=7, min_length=0.002, shoulder_width=0.1, join_width=3):
    mean=sig.mean()
    threshold=sig.std()*deviation_factor
    filtered=np.where(abs(sig - mean) > threshold, np.nan, sig)
    next = 0
    slen=int(shoulder_width*fs)
    max_size=int(join_width*fs)
    mlen=int(min_length*fs)
    tight_bounds=[]
    bounds=[]
    for i in tqdm.tqdm(range(len(filtered))):
        if i < next:
            continue
        if np.isnan(filtered[i]):
            n=0
            while i+n+max_size < len(filtered):
                if not np.isnan(filtered[i+n:i+n+max_size]).any():
                    break
                n+=1
            if np.count_nonzero(np.isnan(filtered[i-slen:i+n+slen]))< mlen:
                break
            filtered[i-slen:i+n+slen] = np.nan
            next = i+n+slen
            tight_bounds.append([i, i+n-1])
            bounds.append([i-slen,i+n+slen-1])
    logger.info("Replace artefacts done, printing summary")
    summary = pd.DataFrame([[s/fs, e/fs] for s,e in bounds], columns=["start", "end"])
    logger.info("Filtering artefacts got the following results\n{}".format(summary))
    return filtered, tight_bounds, bounds

def affine_nan_replace(a):
    nans, x= np.isnan(a), lambda z: z.nonzero()[0]
    ret=a.copy()
    ret[nans]= np.interp(x(nans), x(~nans), a[~nans])
    return ret