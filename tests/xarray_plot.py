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
from matplotlib.backends.backend_pdf import PdfPages

if not __name__=="__main__":
    exit()

xr.set_options(use_flox=True, display_expand_coords=True, display_max_rows=100, display_expand_data_vars=True, display_width=150)
logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
logging.getLogger("flox").setLevel(logging.WARNING)
tqdm.tqdm.pandas(desc="Computing")

MODE: Literal["TEST", "ALL", "SMALL", "BALANCED"]="ALL"
OUT = ["coherence"]
DISPLAY=False


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

signals["SSH"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
signals = signals.set_coords("SSH")
for col in group_index_cols:
    signals[col+"_grp"] =  signals[col].groupby("SSH").map(lambda x:x.isel(Contact=0))
    signals = signals.set_coords(col+"_grp")
signals_sig_type = signals.groupby("sig_type").mean()
signals_sig_type["pwelch_cwt_mean"] = signals_sig_type["pwelch_cwt"].groupby("SSH").mean("Contact")
signals_sig_type["pwelch_spectrogram_mean"] = signals_sig_type["pwelch_spectrogram"].groupby("SSH").mean("Contact")

print(signals_sig_type)

signal_pairs: xr.Dataset = pickle.load(open(cache_path+"signal_pairs_computed.pkl", "rb"))
# signal_pairs = signal_pairs.rename(f3="f2")
signal_pairs["Healthy_1"] = xr.where(signal_pairs["Species_1"]=="Human", 0,  signal_pairs["Healthy_1"])
signal_pairs["Healthy_2"] = xr.where(signal_pairs["Species_2"]=="Human", 0,  signal_pairs["Healthy_2"])
ok = [signal_pairs[f"{col}_1"].equals(signal_pairs[f"{col}_2"]) for col in group_index_cols]
if np.all(ok):
    for col in group_index_cols:
        signal_pairs[col] = signal_pairs[f"{col}_1"]
        signal_pairs=signal_pairs.set_coords(col)
        signal_pairs = signal_pairs.drop(f"{col}_2")
else:
    print(signal_pairs)
    print(ok)
    raise Exception("Problem in data")
    
signal_pairs["SSH"] = xr.DataArray(pd.MultiIndex.from_arrays([signal_pairs[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact_pair'], coords=[signal_pairs["Contact_pair"]])
signal_pairs = signal_pairs.set_coords("SSH")
for col in group_index_cols:
    signal_pairs[col+"_grp"] =  signal_pairs[col].groupby("SSH").map(lambda x:x.isel(Contact_pair=0))
    signal_pairs = signal_pairs.set_coords(col+"_grp")
signal_pairs["sig_type"] = signal_pairs["sig_type_1"] +","+ signal_pairs["sig_type_2"]
signal_pairs["Contact"] = signal_pairs["Contact_1"].astype(str).astype(object) +","+ signal_pairs["Contact_2"].astype(str).astype(object)
signal_pairs = signal_pairs.set_coords(["sig_type", "Contact"])
print(signal_pairs)
signal_pairs_sig_type = signal_pairs.copy()
signal_pairs_sig_type["coherence_wct"] = signal_pairs["coherence_wct"].groupby("sig_type").mean()
signal_pairs_sig_type["coherence_wct_mean"] = signal_pairs_sig_type["coherence_wct"].groupby("SSH").mean("Contact_pair")
signal_pairs_sig_type["coherence_scipy"] = signal_pairs["coherence_scipy"].groupby("sig_type").mean()
# xr.plot.hist(signal_pairs_sig_type["coherence_scipy"])
# plt.show()
signal_pairs_sig_type["coherence_scipy_mean"] = signal_pairs_sig_type["coherence_scipy"].groupby("SSH").mean("Contact_pair")
print(signal_pairs_sig_type)
# exit()

def plot_details(data: xr.DataArray, name=""):
    logger.info(f"plotting details {name}")
    psig_data = data.to_dataframe(name="pwelch").reset_index(allow_duplicates=True)
    pwelch_sig_fig = toolbox.FigurePlot(psig_data, figures="Species", col="SSH", row="sig_type", sharey=False, margin_titles=True, fig_title=name+"_{Species}")
    pwelch_sig_fig.pcolormesh(x="f", y="Contact", value="pwelch", ysort=20.0, ylabels=False)
    return pwelch_sig_fig


def plot_means(data: xr.DataArray, name=""):
    logger.info(f"plotting means {name}")
    pwelch_mean_data = data.to_dataframe(name="pwelch_mean").reset_index(allow_duplicates=True)
    pwelch_mean_fig_structure = toolbox.FigurePlot(pwelch_mean_data, col="Structure_grp", row="sig_type", sharey=False, margin_titles=True, fig_title=f"{name}_structure")
    pwelch_mean_fig_structure.map(sns.lineplot, x="f", y="pwelch_mean", hue="Species_grp", style="Healthy_grp", hue_order=species_order, style_order=condition_order).add_legend()

    pwelch_mean_fig_species = toolbox.FigurePlot(pwelch_mean_data, col="Species_grp", row="sig_type", sharey=False, margin_titles=True, fig_title=f"{name}_species")
    pwelch_mean_fig_species.map(sns.lineplot, x="f", y="pwelch_mean", hue="Structure_grp", style="Healthy_grp", hue_order=structure_order, style_order=condition_order).add_legend()
    return pwelch_mean_fig_structure, pwelch_mean_fig_species

def plot_means(data: xr.DataArray, name=""):
    logger.info(f"plotting means {name}")
    pwelch_mean_data = data.to_dataframe(name="pwelch_mean").reset_index()
    pwelch_mean_fig_structure = toolbox.FigurePlot(pwelch_mean_data, col="Structure_grp", row="sig_type", sharey=False, margin_titles=True, fig_title=f"{name}_structure")
    pwelch_mean_fig_structure.map(sns.lineplot, x="f", y="pwelch_mean", hue="Species_grp", style="Healthy_grp", hue_order=species_order, style_order=condition_order).add_legend()

    pwelch_mean_fig_species = toolbox.FigurePlot(pwelch_mean_data, col="Species_grp", row="sig_type", sharey=False, margin_titles=True, fig_title=f"{name}_species")
    pwelch_mean_fig_species.map(sns.lineplot, x="f", y="pwelch_mean", hue="Structure_grp", style="Healthy_grp", hue_order=structure_order, style_order=condition_order).add_legend()
    return pwelch_mean_fig_structure, pwelch_mean_fig_species


if "pwelch" in OUT:
    pathlib.Path(cache_path+"Figures/pwelch/").mkdir(exist_ok=True, parents=True)
    pp_details = PdfPages(cache_path+"Figures/pwelch/details.pdf")
    plot_details(signals_sig_type["pwelch_cwt"], name="pwelch_wavelet").maximize().save_pdf(pp_details)
    plot_details(signals_sig_type["pwelch_spectrogram"].rename(f2="f"), name="pwelch_spectrogram").maximize().save_pdf(pp_details)
    pp_details.close()
    pp_means = PdfPages(cache_path+"Figures/pwelch/means.pdf")
    for name, data in {
        "all_pwelch_wavelet": signals_sig_type["pwelch_cwt_mean"], 
        "bua_pwelch_wavelet": signals_sig_type["pwelch_cwt_mean"].sel(sig_type="bua"),
        "all_pwelch_spectrogram": signals_sig_type["pwelch_spectrogram_mean"].rename(f2="f"), 
        "bua_pwelch_spectrogram": signals_sig_type["pwelch_spectrogram_mean"].rename(f2="f").sel(sig_type="bua")
    }.items():
        for figs in plot_means(data, name):
            if not DISPLAY:
                figs.maximize().save_pdf(pp_means)
    pp_means.close()

if "coherence" in OUT:
    pathlib.Path(cache_path+"Figures/coherence/").mkdir(exist_ok=True, parents=True)
    pp_details = PdfPages(cache_path+"Figures/coherence/details.pdf")
    r = [plot_details(signal_pairs_sig_type["coherence_wct"], name="coherence_wavelet"),
         plot_details(signal_pairs_sig_type["coherence_scipy"].rename(f2="f"), name="coherence_scipy")
    ]
    if not DISPLAY:
        for figs in r:
            figs.maximize().save_pdf(pp_details)
    pp_details.close()
    pp_means = PdfPages(cache_path+"Figures/coherence/means.pdf")
    for name, data in {
        "all_coherence_wavelet": signal_pairs_sig_type["coherence_wct_mean"],
        "all_coherence_scipy": signal_pairs_sig_type["coherence_scipy_mean"].rename(f2="f")
    }.items():
        for figs in plot_means(data, name):
            if not DISPLAY:
                figs.maximize().save_pdf(pp_means)
    pp_means.close()

if DISPLAY:
    plt.show()


print(signals_sig_type)
