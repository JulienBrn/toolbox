import pandas as pd, numpy as np, functools, xarray as xr, dask.dataframe as dd, dask
import tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from dask.diagnostics import ProgressBar

df = dask.datasets.timeseries(freq="10s")

def f(d: pd.DataFrame):
    print(d)
    test = pd.DataFrame([("a", 0), ("b", 1)], columns=["ex", "val"])
    res = d.merge(test, how="cross")
    res["res"] = res["val"] * res["x"]
    return res

print(df)
print(len(df.index))
df = df.groupby("name").apply(f, meta={"name": str, "id": int, "x": float, "y": float, "ex": str, "val": int, "res":float}).compute()
print(df)
