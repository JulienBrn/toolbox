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
FIGS = ["coherence_mean_figs_structure"]
DISPLAY=False


match MODE:
    case "TEST":
        cache_path = "/media/julien/data1/JulienCache/Test/"
    case "ALL" | "BALANCED":
        cache_path = f"/media/julien/data1/JulienCache/{'All' if MODE=='ALL' else 'Balanced'}/"
    case "SMALL":
        cache_path = "/media/julien/data1/JulienCache/Small/"

group_cols = ["Species", "Structure", "Healthy"]
pair_group_cols = [x+"_1" for x in group_cols] + [x+"_2" for x in group_cols]
species_order = ["Rat", "Monkey", "Human"]
structure_order = ["STN", "GPe", "STR"]
condition_order = [0, 1]
sig_type_order=["bua", "lfp", "spike_times"]

signals: xr.Dataset = pickle.load(open(cache_path+"signals_computed.pkl", "rb"))
signal_pairs: xr.Dataset = pickle.load(open(cache_path+"signal_pairs_computed.pkl", "rb"))
signal_pairs = signal_pairs.where((signal_pairs["FullStructure_1"] != "STN_VMNR") & (signal_pairs["FullStructure_2"] != "STN_VMNR"), drop=True)
# signal_pairs = signal_pairs.sel(sig_preprocessing_pair=[("bua", "neuron_0"), ("bua", "neuron_1"), ("bua", "neuron_2")])


dataset = xr.merge([signals, signal_pairs])
dataset = dataset[[var for var in dataset.variables if ("pwelch" in var) or "coherence" in var]]
scipy_freq_coords = dataset["f2"].to_numpy()
dataset = dataset.interp(f=np.linspace(3, 50, 94, endpoint=False), f2=np.linspace(3, 50, 94, endpoint=False))
for col in dataset.variables:
    if "f2" in dataset[col].dims:
        dataset[col] = dataset[col].rename(f2="f_interp")
    if "f" in dataset[col].dims:
        dataset[col] = dataset[col].rename(f="f_interp")
dataset = dataset.drop(["f", "f2"])
dataset["is_scipy_freq"] = dataset["f_interp"].isin(np.round(scipy_freq_coords*2)/2)
dataset = dataset.set_coords("is_scipy_freq")
dataset["pwelch"] = dataset[["pwelch_cwt", "pwelch_spectrogram"]].to_array(dim="spectral_analysis_function")
dataset.drop_vars(["pwelch_cwt", "pwelch_spectrogram"])
dataset["spectral_analysis_function"] = dataset["spectral_analysis_function"].str.replace("pwelch_", "").str.replace("spectrogram", "scipy.spectrogram").str.replace("cwt", "pycwt.cwt")
coherence = dataset[["coherence_scipy"]].to_array(dim="spectral_analysis_function", name="coherence")
coherence["spectral_analysis_function"] = coherence["spectral_analysis_function"].str.replace("coherence_", "").str.replace("scipy", "scipy.coherence").str.replace("wct", "pycwt.wct")
dataset=xr.merge([dataset, coherence])
dataset = dataset.drop_vars(["coherence_scipy"])



for dim, groupcols in dict(Contact=group_cols, Contact_pair=pair_group_cols, sig_preprocessing=["sig_type"], sig_preprocessing_pair=["sig_type_1", "sig_type_2"]).items():
    grpname = dim+"_grp"
    dataset[grpname] = xr.DataArray(
        pd.MultiIndex.from_arrays([dataset[a].data for a in groupcols],names=groupcols), 
        dims=[dim], coords=[dataset[dim]]
    )
    dataset = dataset.set_coords(grpname)
    # for col in groupcols:
    #     grouped_dataset[col+"_grp"] =  dataset[col].groupby(grpname).first(skipna=True)
    #     grouped_dataset = grouped_dataset.set_coords(col+"_grp")


dataset["coherence_norm"] = np.abs(dataset["coherence"])

grouped_dataset = xr.Dataset()

grouped_dataset["pwelch"] = dataset["pwelch"].groupby("sig_preprocessing_grp").mean().groupby("Contact_grp").mean()
grouped_dataset["coherence"] = dataset["coherence"].groupby("sig_preprocessing_pair_grp").mean().groupby("Contact_pair_grp").mean()
grouped_dataset["coherence_norm"] = dataset["coherence_norm"].groupby("sig_preprocessing_pair_grp").mean().groupby("Contact_pair_grp").mean()
grouped_dataset["n_coherence"] = dataset["coherence"].groupby("sig_preprocessing_pair_grp").mean().groupby("Contact_pair_grp").count("Contact_pair")
grouped_dataset = grouped_dataset.set_coords("n_coherence")
print(grouped_dataset)
# print(np.angle(grouped_dataset["coherence"]))
grouped_dataset["coherence_phase"] = xr.apply_ufunc(np.angle, grouped_dataset["coherence"])
grouped_dataset["coherence_validity"] = np.abs(grouped_dataset["coherence"])/grouped_dataset["coherence_norm"]
grouped_dataset["f_max"] = grouped_dataset["coherence_norm"].sel(f_interp=slice(7, 40)).idxmax("f_interp")

# selected = (
#     grouped_dataset["coherence_norm"].where(grouped_dataset["is_scipy_freq"]).sel(f_interp=slice(7, 40))
#     .where((grouped_dataset["f_interp"] > grouped_dataset["f_max"]-2) & (grouped_dataset["f_interp"] < grouped_dataset["f_max"]+2))
# )
selected = (grouped_dataset["f_interp"] > grouped_dataset["f_max"]-2) & (grouped_dataset["f_interp"] < grouped_dataset["f_max"]+2)
grouped_dataset["coherence_phase"] = grouped_dataset["coherence_phase"].where(selected)

dataset_contact = xr.Dataset()
dataset_contact["pwelch"] = dataset["pwelch"].groupby("sig_preprocessing_grp").mean()
dataset_contact["coherence"] = dataset["coherence"].groupby("sig_preprocessing_pair_grp").mean()
dataset_contact["coherence_norm"] = dataset["coherence_norm"].groupby("sig_preprocessing_pair_grp").mean()
# dataset = dataset.groupby("sig_type_2").mean()
# print(grouped_dataset)

for col in group_cols:
    if grouped_dataset[f"{col}_1"].equals(grouped_dataset[f"{col}_2"]):
        grouped_dataset[col+"(common)"] = grouped_dataset[f"{col}_1"]
        grouped_dataset=grouped_dataset.set_coords(col+"(common)")
    if dataset_contact[f"{col}_1"].equals(dataset_contact[f"{col}_2"]):
        dataset_contact[col+"(common)"] = dataset_contact[f"{col}_1"]
        dataset_contact=dataset_contact.set_coords(col+"(common)")

print(grouped_dataset)
# print(dataset_contact)

figs={}

figs["pwelch_mean_figs_structure"] = lambda: toolbox.FigurePlot(
    data = grouped_dataset["pwelch"].to_dataframe(name="pwelch").dropna(axis="index"),
    figures="spectral_analysis_function", col="Structure", row="sig_type", sharey=False, margin_titles=True, fig_title="mean_pwelch_structure_{spectral_analysis_function}", row_order=sig_type_order,
).map(sns.lineplot, x="f_interp", y="pwelch", hue="Species", style="Healthy", hue_order=species_order, style_order=condition_order).add_legend()

figs["pwelch_mean_figs_species"] = lambda: toolbox.FigurePlot(
    data = grouped_dataset["pwelch"].to_dataframe(name="pwelch").dropna(axis="index", thresh=2),
    figures="spectral_analysis_function", col="Species", row="sig_type", sharey=False, margin_titles=True, fig_title="mean_pwelch_species_{spectral_analysis_function}", row_order=sig_type_order,
).map(sns.lineplot, x="f_interp", y="pwelch", hue="Structure", style="Healthy", 
      hue_order=structure_order, style_order=condition_order, palette=sns.color_palette("dark", n_colors=len(structure_order))
).add_legend()


def quiver(data: pd.DataFrame, x, y, angles, size, color, hue, hue_order: list, zero, **kwargs):
    for h, data in data.groupby(hue):
        # print(h, hue_order.index(h))
        plt.quiver(data[x], data[y], 0.1*data[size]*np.cos(data[angles]+zero), 0.1*data[size]*np.sin(data[angles]+zero), 
               angles="uv", scale=1, scale_units="height",units="height", 
               color=sns.color_palette("tab10", as_cmap=True)(hue_order.index(h)) , **kwargs)
    # colors = data[hue].to_numpy()
    # for i, ho in enumerate(hue_order):
    #     colors = np.where(colors==ho, i, colors)
    # colors=colors.astype(int)
    # print(colors)
    # plt.quiver(data[x], data[y], size*np.cos(data[angles]+zero), size*np.sin(data[angles]+zero), colors,
    #            angles="uv", scale=1, scale_units="height",units="height", cmap = sns.color_palette("tab10", n_colors=len(hue_order), as_cmap=True), **kwargs)

def debug(*args, **kwargs):
    print(kwargs.keys())
    exit()

figs["coherence_mean_figs_structure"] = lambda: (toolbox.FigurePlot(
    data = grouped_dataset[["coherence_norm", "coherence_phase", "coherence_validity"]].where(grouped_dataset["n_coherence"] > 10).to_dataframe().dropna(axis="index", subset="coherence_norm"),
    figures=["spectral_analysis_function", "sig_type_1", "sig_type_2"], col="Structure_1", row="Structure_2", 
    sharey=True, margin_titles=True, fig_title="mean_coherence_structure_{spectral_analysis_function}, {sig_type_1}, {sig_type_2}", 
    row_order=structure_order, col_order=structure_order,
).map(sns.lineplot, x="f_interp", y="coherence_norm", hue="Species(common)", style="Healthy(common)", hue_order=species_order, style_order=condition_order)
.map(quiver,  x="f_interp", y="coherence_norm", angles="coherence_phase", hue="Species(common)", size="coherence_validity", zero=np.pi/2, width=0.005, hue_order=species_order).add_legend()
)

figs["coherence_mean_figs_species"] = lambda: toolbox.FigurePlot(
    data = grouped_dataset[["coherence_norm", "coherence_phase"]].to_dataframe().dropna(axis="index"),
    figures=["spectral_analysis_function", "sig_type_1", "sig_type_2"], col="Species(common)", row="Structure_2", 
    sharey=True, margin_titles=True, fig_title="mean_coherence_structure_{spectral_analysis_function}, {sig_type_1}, {sig_type_2}", 
    row_order=structure_order, col_order=species_order,
).map(sns.lineplot, x="f_interp", y="coherence_norm", hue="Structure_1", style="Healthy(common)", hue_order=structure_order, style_order=condition_order).add_legend()






figs["pwelch_detail_figs"] = lambda: toolbox.FigurePlot(
    data = dataset_contact["pwelch"].to_dataframe(name="pwelch").dropna(axis="index"),
    figures=["spectral_analysis_function", "Species"], col=["Structure", "Healthy"], row="sig_type", sharey=False, margin_titles=True, fig_title="detail_pwelch_{spectral_analysis_function},{Species}", row_order=sig_type_order,
).pcolormesh(x="f_interp", y="Contact", value="pwelch", ysort=20.0, ylabels=False).add_legend()


figs["coherence_detail_figs"] = lambda: toolbox.FigurePlot(
    data = dataset_contact["coherence"].rename({"Healthy(common)":"H(1&2)"}).to_dataframe(name="coherence").dropna(axis="index"),
    figures=["spectral_analysis_function", "Species(common)", "sig_type_1", "sig_type_2"], col=["Structure_1", "H(1&2)"], row="Structure_2", 
    sharey=False, margin_titles=True, fig_title="detail_coherence_{spectral_analysis_function},{Species_1},{sig_type_1}, {sig_type_2}", row_order=structure_order,
).pcolormesh(x="f_interp", y="Contact_pair", value="coherence", ysort=20.0, ylabels=False).add_legend()

if ... in FIGS:
    tmp = set(figs.keys())
    FIGS = tmp - set(FIGS)

for fig in tqdm.tqdm(FIGS, desc="plotting"):
    v: toolbox.FigurePlot =figs[fig]()
    if not DISPLAY:
        v.maximize().save_pdf(f"{cache_path}Figures/{fig}.pdf")


# figs["coherence_detail_figs_structure"] = toolbox.FigurePlot(
#     data = grouped_dataset["coherence"].to_dataframe(name="coherence").dropna(axis="index"),
#     figures=["spectral_analysis_function", "sig_type_1", "sig_type_2"], col="Structure_1", row="Structure_2", sharey=True, margin_titles=True, fig_title="coherence_structure_{spectral_analysis_function}, {sig_type_1}, {sig_type_2}", row_order=structure_order, col_order=structure_order,
# ).map(sns.lineplot, x="f_interp", y="coherence", hue="Species(common)", style="Healthy(common)", hue_order=species_order, style_order=condition_order).add_legend()

# figs["coherence_detail_figs_structure"] = toolbox.FigurePlot(
#     data = grouped_dataset["coherence"].to_dataframe(name="coherence").dropna(axis="index"),
#     figures=["spectral_analysis_function", "sig_type_1", "sig_type_2"], col="Species(common)", row="Structure_2", sharey=True, margin_titles=True, fig_title="coherence_structure_{spectral_analysis_function}, {sig_type_1}, {sig_type_2}", row_order=structure_order, col_order=species_order,
# ).map(sns.lineplot, x="f_interp", y="coherence", hue="Structure_1", style="Healthy(common)", hue_order=structure_order, style_order=condition_order).add_legend()



if DISPLAY:
    plt.show()
exit()
# signals_sig_type["pwelch_cwt_mean"] = signals_sig_type["pwelch_cwt"].groupby("SSH").mean("Contact")
# signals_sig_type["pwelch_spectrogram_mean"] = signals_sig_type["pwelch_spectrogram"].groupby("SSH").mean("Contact")

print(signals_sig_type)


# signal_pairs = signal_pairs.rename(f3="f2")

ok = [signal_pairs[f"{col}_1"].equals(signal_pairs[f"{col}_2"]) for col in group_index_cols]
if np.all(ok):
    for col in group_index_cols:
        signal_pairs[col] = signal_pairs[f"{col}_1"]
        signal_pairs=signal_pairs.set_coords(col)
        signal_pairs = signal_pairs.drop(f"{col}_2")
else:
    # print(signal_pairs)
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
