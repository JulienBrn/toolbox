import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave
from xarray_helper import apply_file_func, auto_remove_dim, nunique


logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
tqdm.tqdm.pandas(desc="Computing")
data_path = pathlib.Path("/home/julien/Documents/all_signals_resampled/")

def extract_unique(a: xr.DataArray, dim: str):
    def get_uniq(a):
        nums = nunique(a, axis=-1, to_str=True)
        if (nums==1).all():
            r = np.take_along_axis(a, np.argmax(~pd.isna(a), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
            return r
        else:
            raise Exception(f"Can not extract unique value. Array:\n {a}\nExample\n{a[np.argmax(nums)]}")
    return xr.apply_ufunc(get_uniq, a, input_core_dims=[[dim]])


signals = xr.open_dataset("signals.nc").load()
metadata = xr.open_dataset("metadata.nc")


signals["time_representation_path"] = xr.apply_ufunc(lambda x: np.where(x=="", np.nan, x), signals["time_representation_path"])
signals = signals.where(signals["Structure"].isin(["GPe", "STN", "STR"]), drop=True)
group_index_cols = ["Species", "Structure", "Healthy"]
signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
signals = signals.set_coords("group_index")

grouped_results = xr.Dataset()
transfert_coords = ["CorticalState", "FullStructure", "Condition"]

for col in transfert_coords:
    grouped_results[col] = signals[col].groupby("group_index").map(extract_unique, dim="Contact").unstack()
grouped_results = auto_remove_dim(grouped_results)
grouped_results = grouped_results.set_coords(transfert_coords)


####### Finally, let's do some stats ##############
grouped_results["n_sessions"] = signals["Session"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda x: np.unique(x).size, a, input_core_dims=[a.dims])).unstack()
grouped_results["ContactCounts"] = signals["has_entry"].any(dim="sig_preprocessing").groupby("group_index").count().unstack()
grouped_results["avg_NeuronCountsPerContact"] = signals["has_entry"].where(~(signals["sig_preprocessing"].isin(["lfp", "bua"])), drop=True).sum(dim="sig_preprocessing").groupby("group_index").mean().unstack()
grouped_results["max_neuron_in_session"] = signals["has_entry"].where(~(signals["sig_preprocessing"].isin(["lfp", "bua"])), drop=True).sum(dim="sig_preprocessing").groupby("group_index").map(lambda a: a.groupby("Session").sum().max()).unstack()


# print(signals["time_representation_path"].to_series().iat[0])

###### Now let's compute stuff on our signals
print(signals)
signals["Duration"] = apply_file_func(lambda arr: arr["index"].max() - arr["index"].min(), ".", signals["time_representation_path"], name="duration", save_group="./durations.pkl")
signals["_diff"] = (signals["Duration"].max("sig_preprocessing") - signals["Duration"].min("sig_preprocessing"))
# grouped_results["DurationDiff"] = 
grouped_results["DurationDiff"] = signals["_diff"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a)[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
grouped_results["DurationDiffBorders"] = signals["_diff"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a)[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
print(grouped_results)
print(np.abs(metadata["Duration"] - signals["Duration"]).max())
# print(signals)


print(grouped_results.drop_vars(["CorticalState", "FullStructure", "Condition"]).to_dataframe().to_string())