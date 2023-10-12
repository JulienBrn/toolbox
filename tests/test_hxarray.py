import pandas as pd, numpy as np, functools, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
import seaborn as sns
import toolbox.hxarray as hx

logger = logging.getLogger(__name__)
beautifullogger.setup()
logging.getLogger("toolbox.hxarray").setLevel(logging.WARNING)



dataset ="all"
# dataset="small"

x: hx.DataArray = pickle.load(open(f"data_hxarray_{dataset}.pickle", "rb"))
def mfunc(a, b):
    # input(a)
    # input(other)
    return a+b
# tmp = hx.apply_ufunc(mfunc, x, aggregated_dims=[["t", "sig_num"]], lowered_dims=[])
# print(tmp)
pwelch = x.mean_tmp({"t", "sig_num"})
print(pwelch)
pwelch2 = x.mean_tmp({"t"}).mean_tmp({"sig_num"})
print(pwelch2)
test = hx.apply_ufunc(lambda a, b: np.abs(a-b).sum(), pwelch, pwelch2, aggregated_dims=[["freq"], ["freq"]], lowered_dims=[])

print(test)
# print(tmp.to_series().to_string())
raise RuntimeError("Stop")
# print(x.to_series().to_string())
pwelch = x.mean_tmp({"t", "sig_num"})
pwelch2 = x.mean_tmp({"t"}).mean_tmp({"sig_num"})
pwelch_df = pwelch.to_series()
pwelch2_df = pwelch2.to_series()
plot_data = pd.DataFrame({"window":pwelch_df, "signal":pwelch2_df})
plot_data.columns.name="Amplitude"
old_names = plot_data.index.names
plot_data = plot_data.stack()
plot_data.name = "Amplitude"
plot_data.index.names = old_names+["avg_unit"]
print(plot_data)
plot_data = plot_data.reset_index()
sns.relplot(kind="line", data=plot_data, x="freq", y="Amplitude", hue="Species", style="avg_unit", row="Structure", col="Healthy")
plt.show()
input(plot_data)
# res_df = res
# with_pd = x.to_series().groupby([col for col in x.a.dims] + ["freq"], observed=True).mean().xs(("Rat", "GPe", False, "bua"))
# print(res_df)
# print(with_pd)
# input()

sp = "Human"
st = "STN"

res = pwelch.sel(Species=sp, Structure=st, Healthy=False, sig_type="bua").to_series().reset_index().sort_values("freq")
plt.plot(res["freq"], res[pwelch.name])
res2 = pwelch.sel(Species=sp, Structure=st, Healthy=True, sig_type="bua").to_series().reset_index().sort_values("freq")
plt.plot(res2["freq"], res2[pwelch.name])
res3 = pwelch2.sel(Species=sp, Structure=st, Healthy=False, sig_type="bua").to_series().reset_index().sort_values("freq")
plt.plot(res3["freq"], res3[pwelch2.name])
res4 = pwelch2.sel(Species=sp, Structure=st, Healthy=True, sig_type="bua").to_series().reset_index().sort_values("freq")
plt.plot(res4["freq"], res4[pwelch2.name])
plt.show()






# logger.info("Converting to pandas")
# p = counts.to_series()
# print(p)
# input()
# print(counts.get_shape_sizes_dict())
# input()
# print(counts)
# embed1 = counts.a.values.flat[0]
# embed2 = embed1.a.values.flat[0]
# print(embed1)
# print(embed2)
# y = x._push_dims_to_one_level_down({"Structure"})
# print(y.get_shape_sizes_dict())
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