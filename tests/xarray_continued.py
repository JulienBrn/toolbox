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

def extract_unique(a: xr.DataArray, dim: str):
    # input(a)
    def get_uniq(a):
        nums = nunique(a, axis=-1, to_str=True)
        if (nums==1).all():
            r = np.take_along_axis(a, np.argmax(~pd.isna(a), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
            if pd.isna(r).any():
                print("RAISING")
                raise Exception(f"nans in r:\n{r}")
            # input(r)
            return r
        else:
            print("RAISING")
            raise Exception(f"Can not extract unique value. Array:\n {a}\nExample\n{a[np.argmax(nums)]}")
    return xr.apply_ufunc(get_uniq, a, input_core_dims=[[dim]])


signals = xr.open_dataset("signals.nc").load()
metadata = xr.open_dataset("metadata.nc")
print(signals)
signals.to_dataframe().to_csv("test.csv")
signals = signals.where(signals["Structure"].isin(["GPe", "STN", "STR"]), drop=True)
group_index_cols = ["Species", "Structure", "Healthy"]
signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
signals = signals.set_coords("group_index")

grouped_results = xr.Dataset()
transfert_coords = ["CorticalState", "FullStructure", "Condition"]
# for k,a in signals["FullStructure"].groupby("group_index"):
#     print(f"Array is {a}")
#     input()
for col in transfert_coords:
    grouped_results[col] = signals[col].groupby("group_index").map(extract_unique, dim="Contact").unstack()
grouped_results = auto_remove_dim(grouped_results)
grouped_results = grouped_results.set_coords(transfert_coords)
grouped_results["Counts"] = signals["time_representation_path"].groupby("group_index").count().unstack()
# grouped_results["CorticalState"] = signals["CorticalState"].groupby("group_index").map(extract_unique, dim="Contact").unstack()
# grouped_results["FullStructure"] = signals["FullStructure"].groupby("group_index").map(extract_unique, dim="Contact").unstack()
print(grouped_results)
print(grouped_results.to_dataframe().to_string())
# grouped_results = xr.Dataset()
# grouped_results = xr.apply_ufunc