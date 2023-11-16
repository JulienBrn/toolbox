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
signals["bua_Duration"] = apply_file_func(lambda arr: arr["index"].max() - arr["index"].min(), ".", signals["time_representation_path"].sel(sig_preprocessing="bua"), name="duration", save_group="./durations.pkl")
signals["spike_duration"] = apply_file_func(lambda arr: float(np.max(arr) - np.min(arr)), ".", signals["time_representation_path"].where(signals["sig_type"] == "spike_times"), name="spike_duration", save_group="./spike_durations.pkl")
signals["n_spikes"] = apply_file_func(lambda arr: arr.size, ".", signals["time_representation_path"].where(signals["sig_type"] == "spike_times"), name="n_spike", save_group="./n_spikes.pkl")
signals["n_spikes/s"] = signals["n_spikes"]/signals["spike_duration"]

signals["_diff"] = (signals["bua_Duration"] - signals["spike_duration"])
print(signals)
signals.to_dataframe().to_csv("tmp.csv")
# # grouped_results["DurationDiff"] = 
grouped_results["DurationDiffBorders"] = signals["_diff"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[-np.inf, -0.001, 0.1, 1, 10, 100, np.inf])[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
grouped_results["NBDurationDiff"] = signals["_diff"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[-np.inf, -0.001, 0.1, 1, 10, 100, np.inf])[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()

grouped_results["n_spikes_borders"] = signals["n_spikes"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 5, 20, 50, 100, 500, np.inf])[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
grouped_results["NB_n_spikes"] = signals["n_spikes"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 5, 20, 50, 100, 500, np.inf])[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
grouped_results["n_spikes/s_borders"] = signals["n_spikes/s"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 1, 5, 10, 20, 50, np.inf])[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
grouped_results["NB_n_spikes/s"] = signals["n_spikes/s"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 1, 5, 10, 20, 50, np.inf])[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
# print(grouped_results)
# print(np.abs(metadata["Duration"] - signals["Duration"]).max())
# # print(signals)
# errors = signals.where(np.abs(signals["_diff"]) >10, drop=True).merge(metadata["signal_file_source"], join="left").to_dataframe()
# print(errors)
# errors.to_csv("errors.csv")
# print(errors.loc[errors["signal_file_source"]=="File(CTL/A1/20051207/a07/GPe/Raw.mat)[pjx301a_Probe2, values]", :])
# print(metadata)
print(grouped_results)

print(grouped_results.drop_vars(["CorticalState", "FullStructure", "Condition"]).to_dataframe().to_string())