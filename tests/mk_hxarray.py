import pandas as pd, numpy as np, functools, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger

logger = logging.getLogger(__name__)
beautifullogger.setup()

data_path = pathlib.Path(f"/home/julien/Documents/spectrogram_export/database.tsv")
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
        r.name = "spectrogram"
        return toolbox.hxarray.DataArray.from_xarray(r)
    d= [mk_hxarray(df) for df in d]
    d = pd.Series(d)
    d.index.name = "sig_num"
    d = xr.DataArray.from_series(d)
    d = toolbox.hxarray.DataArray.from_xarray_level(d, "signal")
    # input(d.name)
    return d

tqdm.tqdm.pandas(desc="Loading data files")
init_cols = list(set(df.columns) -  {"path", "suffix", "column", "Ressource", "fullpath"})
for col in init_cols:
    if df[col].dtype == object:
        df[col] = df[col].astype("category")
df.set_index(init_cols, inplace=True)
x = df["fullpath"].progress_apply(load_data)
x = xr.DataArray.from_series(x)
x = toolbox.hxarray.DataArray.from_xarray_level(x, "metadata")
x.a = x.a.transpose("Species", "Structure", "Healthy", "signal_resampled_type", ...)
x.a = x.a.rename(signal_resampled_type="sig_type")
x.name = "spectrogram_all"
logger.info("Dumping to file")
pickle.dump(x, open("data_hxarray_all.pickle", "wb"))
logger.info("Dumped")
logger.info("Computing small dataset")
x = x.sel(sig_num=[1, 2, 3], t=[0.5, 0.55])
x.name = "spectrogram_small"
logger.info("Exporting small")
pickle.dump(x, open("data_hxarray_small.pickle", "wb"))
logger.info("Dumped")