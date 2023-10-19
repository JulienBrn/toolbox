import pandas as pd, numpy as np, functools, xarray as xr, dask.dataframe as dd, dask
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from dask.diagnostics import ProgressBar
import graphchain

logger = logging.getLogger(__name__)
beautifullogger.setup()
logging.getLogger("fsspec.local").setLevel(logging.WARNING)
logging.getLogger("toolbox.hxarray").setLevel(logging.WARNING)
import warnings
warnings.simplefilter('ignore',lineno=109)
ProgressBar().register()

all_index_cols=["Species", "Structure", "Healthy", "sig_type",  "sig_num", "t", "freq", "agg_type", "band_name", "kde_bandwith"]
def gcols(l, df):
    return [col for col in all_index_cols if not col in l and col in df.columns]

species_order = ["Rat", "Monkey", "Human"]
structure_order=["STN", "GPe", "STR"]
sig_type_order=["lfp", "bua", "spikes"]
condition_order = ["Park", "Control"]

nwindows,nsignals = 5, 2
# nwindows,nsignals = None, None
if (nwindows,nsignals) == (None,None):
    spectrogram_path = pathlib.Path(f"./Dataset/database_spectrogram_all.parquet")
else:
    spectrogram_path = pathlib.Path(f"./Dataset/database_spectrogram_{nwindows}, {nsignals}.parquet")
if not spectrogram_path.exists():
    raise Exception(f"Spectrogram data at path {spectrogram_path} unfound")
spectrogram = dd.read_parquet(spectrogram_path)
bands_df = pd.DataFrame([[8, 15], [16, 30], [8, 30], [31, 49]], columns=["band_start","band_end"]).merge(pd.Series(["median", "mean"], name="band_agg"), how="cross")
bands_df["band_name"] = bands_df.apply(lambda row: f"band({row['band_start']}, {row['band_end']}, agg={row['band_agg']})", axis=1)
bands_df["band_bounds"] = bands_df.apply(lambda row: f"{row['band_start']}, {row['band_end']}", axis=1)
bands_df.set_index("band_name")
# input(bands_df)

# def aggregate_by_apply(self, by, f, vectorize_over = [], *args, **kwargs):
#     global all_index_cols
#     keys = [col for col in all_index_cols if not col in by and col in self.columns]
#     vectorize_over = set(vectorize_over).intersection(set(keys))
#     if vectorize_over:
#         keys = [k for k in keys if not k in vectorize_over]
#         f = 
#     return self.groupby(keys, *args, **kwargs, observed=True).apply(f)

def aggregate_by(self, by, *args, **kwargs):
    global all_index_cols
    keys = [col for col in all_index_cols if not col in by and col in self.columns]
    return self.groupby(keys, *args, **kwargs, observed=True)

dd.DataFrame.aggregate_by = aggregate_by
pd.DataFrame.aggregate_by = aggregate_by

def apply_index(self: pd.DataFrame):
    return self.set_index([col for col in all_index_cols if col in self.columns], append=True)
pd.DataFrame.apply_index = apply_index

def new_str(self):
    old = self.old_str()
    addition = f"Index names = {self.index.names}\nColumns = {self.columns}"
    return f"{old}\n{addition}"
pd.DataFrame.old_str = pd.DataFrame.__str__
pd.DataFrame.__str__ = new_str

def merge(a, b):
    on_a = set(a.index.names).union(set(a.columns)).intersection(set(all_index_cols))
    on_b = set(b.index.names).union(set(b.columns)).intersection(set(all_index_cols))
    on = list(on_a.intersection(on_b))
    # print(a)
    # print(b)
    # input(on)
    return a.reset_index().merge(b.reset_index(), how="outer", on=on).apply_index().reset_index(level=0, drop=True)


class Autosave:
    def __init__(self, name, version, debug=False):
        if (nwindows,nsignals) == (None,None):
            self.result_path = pathlib.Path(f"./Results/{name}/Data/from_spectrogram_all, {version}.pickle")
        else:
            self.result_path = pathlib.Path(f"./Results/{name}/Data/from_spectrogram_{nwindows}, {nsignals}, {version}.pickle")
        self.name=name
        self.debug = debug
    def __call__(self, f):
        def new_f(*args, **kwargs):
            if not self.result_path.exists():
                logger.info(f"Computing {self.name}")
                res = f(*args, **kwargs)
                if isinstance(res, dd.DataFrame) or isinstance(res, dd.Series):
                    logger.info(f"{self.name} defined as a dask element." + "" if not self.debug else " Computing extract...")
                    if self.debug:
                        logger.info(f"Extract is\n{res.head(5)}")
                    return res
                elif not self.debug:
                    logger.info(f"{self.name} computed, now saving")
                    self.result_path.parent.mkdir(exist_ok=True, parents=True)
                    pickle.dump(res, self.result_path.open("wb"))
                    logger.info(f"{self.name} saved")
            else:
                logger.info(f"Loading {self.name}")
                res= pickle.load(self.result_path.open("rb"))
            logger.info(f"{self.name} is\n{res}")
            if isinstance(res, pd.DataFrame):
                self.result_path.parent.mkdir(exist_ok=True, parents=True)
                res.to_csv(self.result_path.with_suffix(".tsv"), sep="\t")
            return res
        return new_f
    

@Autosave("counts", version = 7)
def get_counts():
    spectrogram_sig_count = spectrogram.aggregate_by(["t", "sig_num"])["sig_num"].apply(lambda d: d.nunique()).to_frame().compute().rename(columns={"sig_num":"sig_count"}).apply_index()
    spectrogram_window_count = spectrogram.aggregate_by(["t", "sig_num"])["amp"].count().to_frame().compute().rename(columns={"amp":"window_count"}).apply_index()
    return merge(spectrogram_sig_count, spectrogram_window_count)

@Autosave("pwelch", version = 4)
def get_pwelch():
    pwelch_window_median: pd.DataFrame = spectrogram.aggregate_by(["sig_num", "t"])["amp"].median().to_frame().compute().assign(agg_type = "median(t, sig_num)").apply_index()
    pwelch_window_avg: pd.DataFrame = spectrogram.aggregate_by(["sig_num", "t"])["amp"].mean().to_frame().compute().assign(agg_type = "mean(t, sig_num)").apply_index()
    
    pwelch_sig_median: pd.DataFrame = spectrogram.aggregate_by(["t"])["amp"].median().to_frame().reset_index().aggregate_by(["sig_num"])["amp"].median().to_frame().compute().assign(agg_type = "median(sig_num, median(t))").apply_index()
    pwelch_sig_mean: pd.DataFrame = spectrogram.aggregate_by(["t"])["amp"].mean().to_frame().reset_index().aggregate_by(["sig_num"])["amp"].mean().to_frame().compute().assign(agg_type = "mean(sig_num, mean(t))").apply_index()

    pwelch = pd.concat([pwelch_window_median, pwelch_window_avg, pwelch_sig_median, pwelch_sig_mean]).reset_index()
    return pwelch


def plot_pwelch(pwelch, counts):
    pwelch = merge(pwelch, counts).reset_index()
    pwelch=pwelch.loc[pwelch["freq"] < 45].copy()
    pwelch["window_count"] = pwelch["window_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    pwelch["sig_count"] = pwelch["sig_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    pwelch["Condition"] = pwelch["Healthy"].apply(lambda x: "Control" if x else "Park")
    window_size_order = pwelch["window_count"].drop_duplicates().sort_values(ascending=False).to_list()
    sig_sizes = pwelch["sig_count"].drop_duplicates().sort_values(ascending=False).to_list()

    pwelch_figs = toolbox.FigurePlot(data=pwelch, 
        figures="sig_type", fig_title="Pwelch sig_type={sig_type}", 
        col="agg_type", row="Structure", row_order=structure_order, 
        aspect=2, margin_titles=True
    )
    pwelch_figs.map(sns.scatterplot, x="freq", y="amp",  
        hue="Species", hue_order = species_order, 
        size="window_count", size_order = window_size_order, 
        edgecolor="black"
    )
    pwelch_figs.map(sns.lineplot, x="freq", y="amp",  
        hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
        size="sig_count", size_order=sig_sizes
    )
    pwelch_figs.tight_layout().add_legend()

def plot_kde(kde):
    kde = merge(kde, bands_df).reset_index()
    if not "kde_bandwith" in kde.columns:
        kde["kde_bandwith"] = np.round(kde["bandwith"], 5)
    kde["Condition"] = kde["Healthy"].apply(lambda x: "Control" if x else "Park")
    # line_sizes = {x:x*10**4 for x in kde["kde_bandwith"].drop_duplicates()}
    figs = toolbox.FigurePlot(data= kde, 
        figures=["sig_type", "band_agg", "kde_bandwith"], 
        fig_title="Gaussian Kernel Density Estimate, sig_type={sig_type}, band_agg={band_agg}, kde_bandwith={kde_bandwith}", 
        row="Structure", row_order=structure_order,
        col="band_bounds",  col_order=bands_df["band_bounds"].drop_duplicates(),
        aspect=2, margin_titles=True
    )
    figs.map(sns.lineplot, x="amp", y="density",  
        hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
        # size="kde_bandwith", sizes=line_sizes
    )
    figs.tight_layout().add_legend()

@Autosave("band_spectrogram", version = 4)
def get_band_df():
    def mk_band(d):
        d = d.merge(bands_df, how="cross")
        if d["freq"].count()>20:
            d = d[(d["freq"] >= d["band_start"]) & (d["freq"] <= d["band_end"])]
        res = d.aggregate_by("freq").apply(lambda d: d["amp"].median() if d["band_agg"].iat[0] == "median" else d["amp"].mean() if d["band_agg"].iat[0] == "mean" else np.nan)
        res = pd.DataFrame({"amp":res}).reset_index()
        return res
    return spectrogram.aggregate_by(["t", "freq"]).apply(mk_band).reset_index(drop=True)

@Autosave("band_kde", version = 14)
def get_kde(bands: dd.DataFrame):
    def compute_kde(d):
        if len(d) < 2:
            r = lambda x: np.zeros_like(x)
        else:
            import scipy
            try:
                r = scipy.stats.gaussian_kde(d["amp"].values, bw_method=d["kde_bandwith"].iat[0]/d.values.std())
            except:
                print(d)
                raise
        res = pd.DataFrame()
        res["amp"] = np.linspace(0, 0.010, 200)
        res["density"] = r(res["amp"].values)
        res["bandwith"] = (r.covariance_factor() * d.values.std()) if len(d) >= 2 else np.nan
        # res["kde_bandwith"] = d["kde_bandwith"].iat[0]
        res.set_index("amp")
        return res
    df = dd.multi.merge(bands.assign(__tmp=0), pd.Series([10**(-i) for i in range(2, 6)], name="kde_bandwith").to_frame().assign(__tmp=0), on="__tmp")
    kde = df.aggregate_by(["sig_num", "t"])[["amp", "kde_bandwith"]].apply(compute_kde).reset_index().compute()
    return kde


@Autosave("sorted_bands", version = 2, debug=True)
def get_sorted_bands(bands):
    r:dd.DataFrame = bands.aggregate_by(["sig_num", "t"])["amp"].apply(lambda d: d.values if len(d) >5 else np.zeros(2), meta={"amp":object})
    print("yes:", r.dtypes)
    raise Exception("Stop")
    return r

@Autosave("relative_distributions", version = 3, debug=True)
def get_relative_distributions(sorted_bands):
    def rel_dist(d: pd.Series):
        logger.info(f"{type(d['amp'])}\n{d}")
        a,b = d["amp"], d["other_amp"]
        if not isinstance(a, np.ndarray):
            return np.zeros(2)
        else:
            return np.searchsorted(a, b)

    grp_cols = gcols(["Healthy", "band_name"], sorted_bands)
    sorted_bands = sorted_bands.merge(bands_df, on="band_name")
    all = dd.multi.merge(sorted_bands, 
        sorted_bands.rename(columns=lambda c: f"other_{c}" if not c in grp_cols else c), on=grp_cols).reset_index()
    # input(all.compute()[["band_name", "other_band_name"]])
    keep_agg = (all["band_agg"] == all["other_band_agg"]) 
    rm_identical = (all["band_name"] == all["other_band_name"]) | (all["Healthy"]==all["other_Healthy"]) & all["Healthy"]
    all = all.loc[keep_agg & (~rm_identical)]
    # input(all.compute()[["band_name", "other_band_name"]])
    all["rel_dist"] = all.apply(rel_dist, axis=1)
    input(all.compute())


counts: pd.DataFrame = get_counts()
pwelch: pd.DataFrame = get_pwelch()
bands: dd.DataFrame  = get_band_df()
kde: pd.DataFrame = get_kde(bands)
# sorted_bands = get_sorted_bands(bands)
# relative_distributions=get_relative_distributions(sorted_bands)
# raise Exception("Stop")



plot_pwelch(pwelch, counts)
plot_kde(kde)
plt.show()

# # @dask.delayed
# def load_spectrograms(path):
#     return dd.read_parquet(path)

# # @dask.delayed
# def compute_pwelch(spectrogram):
#     logger.info("Computing pwelch")
#     print("Computing")
#     pwelch = spectrogram.aggregate_by(["t", "sig_num"])["amp"].mean(numeric_only=True)
#     return pwelch

# # pwelch : dd.DataFrame= dd.read_parquet("all_pwelch.parquet")
# # pwelch = pwelch.compute()
# # print(pwelch)
# # raise Exception("Stop")


# with dask.config.set(): #delayed_optimize=graphchain.optimize
#     all_df = load_spectrograms("database_spectrogram_all.parquet")
#     print(all_df.columns)
#     # logger.info(f"All df contains {len(all_df.index)} entries. A Snapshot is\n{all_df.head(10)}")
#     pwelch = compute_pwelch(all_df).to_frame()
#     print(type(pwelch))
#     pwelch.to_parquet("all_pwelch.parquet")
#     print(pwelch)
#     logger.info(f"pwelch computed, length= {len(pwelch.index)}")

