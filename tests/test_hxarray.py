import pandas as pd, numpy as np, functools, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger

logger = logging.getLogger(__name__)
beautifullogger.setup()

logger.info("Loading data")
x = pickle.load(open("data_hxarray.pickle", "rb"))
logger.info("Loaded")
print(x)
y = x._push_dim_to_one_level_down("Structure")
print(y)
raise RuntimeError("Stop")
print(x.count())
x = x.sel(signal_num=[1, 2, 3])
print(x)
print(x.count())






raise RuntimeError("Stop")







result: Literal["spectrogram"] = "spectrogram"
# target = (["Monkey"], "real")
data_path = pathlib.Path(f"/home/julien/Documents/{result}_export/database.tsv")
df: pd.DataFrame = toolbox.df_loader.load(data_path)

df["fullpath"] = df["path"] + df["suffix"]
def load_data(path:pathlib.Path, n=None):
    d: List[pd.DataFrame] = toolbox.pickle_loader.load(pathlib.Path("/home/julien/Documents/" ) / pathlib.Path(path).relative_to("/home/julien/Documents/GUIAnalysis/"))
    def mk_hxarray(df):
        df.index.name = "t"
        df.columns.name=("freq")
        s= pd.Series(df.stack())
        # s.index.names=["t", "freq"]
        r = xr.DataArray.from_series(s)
        return toolbox.hxarray.DataArray.from_xarray(r)
    d= [mk_hxarray(df) for df in d]
    # print(d)
    d = pd.Series(d)
    d.index.name = "signal_num"
    d = xr.DataArray.from_series(d)
    # print(d)
    d = toolbox.hxarray.DataArray.from_xarray_level(d, None)
    # print(d)
    # input()
    # if "test" in target[1]:
    #     s, e = (2, 4)
    #     d= [df.iloc[0:np.random.randint(s, e), :].copy() for k, df in enumerate(d)]
    # res = pd.concat({i:df for i, df in enumerate(d)}, names=["signal", "window"])
    # ar = xr.Dataset.from_dataframe(res).to_array("freq").stack(window_id = ["signal", "window"]).dropna(dim="window_id", how="all")
    # eff = float(ar.count()/ar.size)
    # print(ar, "\nefficiency:", eff)
    # # input()
    # return pd.Series({"Data" : Toto(ar), "efficiency" : eff, "win_id_size": ar["window_id"].size})
    return d

tqdm.tqdm.pandas(desc="Loading data files")
# df = df[(df["Species"].isin(target[0]))].copy()
init_cols = list(set(df.columns) -  {"path", "suffix", "column", "Ressource", "fullpath"})
for col in init_cols:
    if df[col].dtype == object:
        df[col] = df[col].astype("category")
df.set_index(init_cols, inplace=True)
x = df["fullpath"].progress_apply(load_data)
# print(x)
x = xr.DataArray.from_series(x)
# print(x)
x = toolbox.hxarray.DataArray.from_xarray_level(x, "metadata")
print(x)
x = x.sel(signal_num=[1, 2, 3])
input(x)
pickle.dump(x, open("data_hxarray.pickle", "wb"))
# df[["Data", "efficiency", "win_id_size"]] = x
# df.drop(columns=["path", "suffix", "fullpath", "column", "Ressource"], inplace=True)
# initial_cols = list(set(df.columns) - set(["Data", "efficiency", "win_id_size"]))