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

def replace_artefacts_with_nans(sig, fs, deviation_factor=7, min_length=0.002, shoulder_width=0.2, join_width=0.5):
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


def get_bad_ranges(sig: np.array, deviation_factor):
    mean= np.nanmean(sig)
    std = np.nanstd(sig)
    # logger.debug("mean={}, std={}".format(mean, std))

    mask = np.where(np.isnan(sig) | (abs(sig - mean) > deviation_factor*std), 1, 0)
    changes = mask[:-1] - mask[1:]
    start_indices = np.where(changes==-1)[0]+1
    end_indices = np.where(changes==1)[0]+1
    
    if mask[0] == 1:
        start_indices= np.insert(start_indices,0,0)
    if mask[-1] == 1:
        end_indices = np.append(end_indices, len(mask))

    return start_indices, end_indices


def replace_artefacts_with_nans2(sig, fs, deviation_factor, min_length, join_width, recursive, shoulder_width):
    tmp_sig = sig.copy().astype(float)
    old_start_indices=np.array([])
    old_end_indices=np.array([])
    nb_rec=0
    while True:
      logger.debug("Going into recursion {}".format(nb_rec))
      start_indices, end_indices = get_bad_ranges(tmp_sig, deviation_factor)
      df = pd.DataFrame()
      df["start"] = start_indices
      df["end"] = end_indices
      df["Kept"] = df["end"] - df["start"] >=  min_length*fs
      df.loc[df["Kept"], "Merged_before"] = df.loc[df["Kept"], "start"] - df.loc[df["Kept"], "end"].shift(1) < join_width*fs
      df.loc[df["Kept"], "Merged_after"] = df.loc[df["Kept"], "start"].shift(-1) - df.loc[df["Kept"], "end"] < join_width*fs
      start_indices = df.loc[df["Kept"] & (df["Merged_before"] == False), "start"].to_numpy(copy=True)
      end_indices = df.loc[df["Kept"] & (df["Merged_after"] == False), "end"].to_numpy(copy=True)
      # logger.debug("Df\n{}".format(df.to_string()))
      logger.debug("Start end list\n{}".format("\n".join([str(t) for t in zip(start_indices, end_indices)])))
      if not recursive:
          break
      if np.array_equal(start_indices, old_start_indices) and np.array_equal(end_indices, old_end_indices):
          break
      nb_rec+=1
      old_start_indices = start_indices
      old_end_indices = end_indices
      for s,e in zip(start_indices, end_indices):
        tmp_sig[int(max(0,int(s))):int(min(len(sig), int(e)))] = np.nan


    # df.loc[df["Kept"], "Merged_before"] = df.loc[df["Kept"], "start"] - df.loc[df["Kept"], "end"].shift(1) < join_width*fs
    # df.loc[df["Kept"], "Merged_after"] = df.loc[df["Kept"], "start"].shift(-1) - df.loc[df["Kept"], "end"] < join_width*fs
    # logger.debug("DF:\n{}".format(df.to_string()))
    # start_indices = df.loc[df["Kept"] & (df["Merged_before"] == False), "start"].to_numpy(copy=True)
    # end_indices = df.loc[df["Kept"] & (df["Merged_after"] == False), "end"].to_numpy(copy=True)
    # logger.debug("Tight indices:\n{}".format("\n".join([str(t) for t in zip(start_indices, end_indices)])))
    
    # input("wait")
    # # keptdf["Merged"] = keptdf[""]
    # # self._d.loc[self._d["event_name"]==name, "old_value"]=self._d.loc[self._d["event_name"]==name, "value"].shift(1)

    # min_length_mask = (end_indices - start_indices) >= min_length*fs
    # start_indices = start_indices[min_length_mask]
    # end_indices = end_indices[min_length_mask]


    # df["length"]
    # df["Decision"] = "Kept"
    # df.loc[~df["start"].isin(start_indices), "Decision"] = "Discarded"
    # # df["filt_end"] = end_indices
    # logger.debug("[Stage 1] Potential artefacts df\n{}".format(df.to_string()))

    # if len(start_indices) > 0:
    #   merge_mask = np.where(start_indices[1:] - end_indices[:-1] < join_width*fs, 1, 0)
    #   start_indices = start_indices[np.insert(np.where(merge_mask == 0)[0]+1, 0, 0)]
    #   end_indices = end_indices[np.append(np.where(merge_mask == 0)[0], len(end_indices)-1)]

    # df.loc[((~df["start"].isin(start_indices))
    #        & (df["Decision"] == "Kept"))
    #        , "Decision"] = "Merged"
    # # df["merged_end"] = end_indices

    # start_indices -= int(shoulder_width*fs)
    # end_indices += int(shoulder_width*fs)

    # df["shoulder_start"] = df["start"] - int(shoulder_width*fs)
    # df["shoulder_end"] = df["end"] + int(shoulder_width*fs)
    # df["shoulder_start"] = start_indices
    # df["shoulder_end"] = end_indices
    # logger.debug("[Stage 2] Potential artefacts df\n{}".format(df.loc[df["Decision"] != "Discarded", ["start", "end", "Decision"]].to_string()))
    start_indices -= int(shoulder_width*fs)
    end_indices += int(shoulder_width*fs)
    filtered = sig.copy().astype(float)
    for s,e in zip(start_indices, end_indices):
      filtered[int(max(0,int(s))):int(min(len(filtered), int(e)))] = np.nan
    
    # logger.debug("Start end list\n{}".format("\n".join([str(t) for t in zip(start_indices, end_indices)])))
    # if len(start_indices) != len(end_indices) or len(start_indices) != len(df.loc[df["Decision"] == "Kept", "Decision"]):
    #     logger.error("Problem with nb artefacts")
    # else:
    #     logger.info("lengths ok")
    # logger.debug("Kept are\n{}".format(df.loc[df["Decision"] == "Kept", ["start", "end"]]))
    # input("waiting")

    return filtered, {}, [[s, e] for s, e in zip(start_indices, end_indices)]
        


def affine_nan_replace(a):
    nans, x= np.isnan(a), lambda z: z.nonzero()[0]
    ret=a.copy()
    ret[nans]= np.interp(x(nans), x(~nans), a[~nans])
    return ret