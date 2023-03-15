import numpy as np

def remove_artefacts(signal, factor=2) -> None:
    """
    Removes artefacts from signal by removing recursively all values signal[i] such as 
    abs(signal[i-1] * factor) < signal[i] and abs(signal[i+1] * factor) < signal[i]
    """
    for i in range(1, len(signal)-1):
        if abs(signal[i-1] * 2) < abs(signal[i]) and abs(signal[i+1] * 2) < abs(signal[i]):
            signal[i]=np.nan
            i=i-1