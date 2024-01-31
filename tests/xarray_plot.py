import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave
from xarray_helper import apply_file_func, auto_remove_dim, nunique, apply_file_func_decorator, extract_unique
import scipy.signal

if not __name__=="__main__":
    exit()

xr.set_options(use_flox=True, display_expand_coords=True, display_max_rows=100, display_expand_data_vars=True, display_width=150)
logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
logging.getLogger("flox").setLevel(logging.WARNING)
tqdm.tqdm.pandas(desc="Computing")

MODE: Literal["TEST", "ALL", "SMALL", "BALANCED"]="ALL"
match MODE:
    case "TEST":
        cache_path = "/media/julien/data1/JulienCache/Test/"
    case "ALL" | "BALANCED":
        cache_path = f"/media/julien/data1/JulienCache/{'All' if MODE=='ALL' else 'Balanced'}/"
    case "SMALL":
        cache_path = "/media/julien/data1/JulienCache/Small/"

group_index_cols = ["Species", "Structure", "Healthy"]
species_order = ["Rat", "Monkey", "Human"]
structure_order = ["STN", "GPe", "STR"]
condition_order = [0, 1]

signals: xr.Dataset = pickle.load(open(cache_path+"signals_computed.pkl", "rb"))
# signals["SSH"] = signals["Species"] + signals["Structure"] + signals["Healthy"].astype(str)
# signals = signals.groupby("SSH").map(lambda x:x)
# print(signals)
# exit()
# signals["Species"] = signals["Species"].groupby("group_index").map(lambda x:x.isel(Contact=1))
# signals["Structure"] = signals["Structure"].groupby("group_index").map(lambda x:x.isel(Contact=1))
# signals["Healthy"] = signals["Healthy"].groupby("group_index").map(lambda x:x.isel(Contact=1))
# signals["pwelch_cwt_mean"] = signals["pwelch_cwt"].groupby("group_index").mean("Contact")
signals["SSH"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
signals = signals.set_coords("SSH")
for col in group_index_cols:
    signals[col+"_grp"] =  signals[col].groupby("SSH").map(lambda x:x.isel(Contact=0))
    signals = signals.set_coords(col+"_grp")
signals_sig_type = signals.groupby("sig_type").mean()
signals_sig_type["pwelch_cwt_mean"] = signals_sig_type["pwelch_cwt"].groupby("SSH").mean("Contact")

print(signals_sig_type)




# psig_data = signals_sig_type["pwelch_cwt"].to_dataframe().reset_index()
# pwelch_sig_fig = toolbox.FigurePlot(psig_data, figures="Species", col="SSH", row="sig_type", sharey=False, margin_titles=True, fig_title="pwelch_{Species}")
# pwelch_sig_fig.pcolormesh(x="f", y="Contact", value="pwelch_cwt", ysort=20.0, ylabels=False)

# pwelch_mean_data = signals_sig_type["pwelch_cwt_mean"].to_dataframe().reset_index()
# pwelch_mean_fig = toolbox.FigurePlot(pwelch_mean_data, col="Structure_grp", row="sig_type", sharey=False, margin_titles=True, fig_title="pwelch_structure")
# pwelch_mean_fig.map(sns.lineplot, x="f", y="pwelch_cwt_mean", hue="Species_grp", style="Healthy_grp", hue_order=species_order, style_order=condition_order).add_legend()

# pwelch_mean_fig = toolbox.FigurePlot(pwelch_mean_data, col="Species_grp", row="sig_type", sharey=False, margin_titles=True, fig_title="pwelch_species")
# pwelch_mean_fig.map(sns.lineplot, x="f", y="pwelch_cwt_mean", hue="Structure_grp", style="Healthy_grp", hue_order=structure_order, style_order=condition_order).add_legend()


pwelch_mean_data_bua = signals_sig_type["pwelch_cwt_mean"].sel(sig_type="bua").to_dataframe().reset_index()

pwelch_mean_fig_bua = toolbox.FigurePlot(pwelch_mean_data_bua, col="Structure_grp", row="sig_type", sharey=False, margin_titles=True, fig_title="pwelch_structure_bua")
pwelch_mean_fig_bua.map(sns.lineplot, x="f", y="pwelch_cwt_mean", hue="Species_grp", style="Healthy_grp", hue_order=species_order, style_order=condition_order).add_legend()

pwelch_mean_fig_bua = toolbox.FigurePlot(pwelch_mean_data_bua, col="Species_grp", row="sig_type", sharey=False, margin_titles=True, fig_title="pwelch_species_bua")
pwelch_mean_fig_bua.map(sns.lineplot, x="f", y="pwelch_cwt_mean", hue="Structure_grp", style="Healthy_grp", hue_order=structure_order, style_order=condition_order).add_legend()

# def test(a):
#     print(a)
#     r = scipy.signal.find_peaks(a)
#     print(r)
#     input()
#     return a, 5

# r, _= xr.apply_ufunc(test, signals["pwelch_cwt_mean"], input_core_dims=["f"], output_core_dims=[["peak"], []], vectorize=True)

# # signals["pwelch_cwt_mean"].max("f")
# print(r)
plt.show()
print(signals_sig_type)
exit()

psig_data = signals["pwelch_cwt"].to_dataframe().reset_index()
psig_data = psig_data[(psig_data["f"] > 6) & (psig_data["f"] < 45)].copy()
psig_data["SSH"] = psig_data["Species"].astype(str) + psig_data["Structure"].astype(str) + psig_data["Healthy"].astype(str)
psig_data = psig_data.groupby(["Contact", "f", "sig_type", "SSH", "Species", "Structure", "Healthy"])["pwelch_cwt"].mean().reset_index()
psig_data["sig_type"] = np.where(psig_data["sig_type"].str.contains("spike"), "spike", psig_data["sig_type"])
# psig_data = psig_data.sort_values("pwelch")
# print(psig_data["SSH"])
# psig_data = psig_data[psig_data["SSH"].isin(["RatGPe0"])]
pwelch_sig_fig = toolbox.FigurePlot(psig_data, figures="Species", col="SSH", row="sig_type", sharey=False, margin_titles=True)
pwelch_sig_fig.pcolormesh(x="f", y="Contact", value="pwelch_cwt", ysort=20.0, ylabels=False)

pwelch_mean_data = signals["pwelch_cwt_mean"].to_dataframe().reset_index()
pwelch_mean_fig = toolbox.FigurePlot(pwelch_mean_data, figures="Species", col="SSH", row="sig_type", sharey=False, margin_titles=True)
pwelch_mean_fig.map(sns.relplot, x="f", y="pwelch_cwt_mean")

plt.show()