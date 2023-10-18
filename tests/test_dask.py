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
ProgressBar().register()

all_index_cols=["Species", "Structure", "Healthy", "sig_type",  "signal_num", "t", "freq", "agg_type", "band_name"]
nwindows,nsignals = 500, 50
# nwindows,nsignals = None, None
if (nwindows,nsignals) == (None,None):
    spectrogram_path = pathlib.Path(f"./Dataset/database_spectrogram_all.parquet")
else:
    spectrogram_path = pathlib.Path(f"./Dataset/database_spectrogram_{nwindows}, {nsignals}.parquet")
if not spectrogram_path.exists():
    raise Exception(f"Spectrogram data at path {spectrogram_path} unfound")
spectrogram = dd.read_parquet(spectrogram_path)
bands_df = pd.DataFrame([[8, 15, "median"], [16, 30, "median"], [8, 30, "median"], [31, 49, "median"]], columns=["band_start","band_end", "band_agg"])
bands_df["band_name"] = bands_df.apply(lambda row: f"band({row['band_start']}, {row['band_end']}, agg={row['band_agg']})", axis=1)
bands_df.set_index("band_name")

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
                    logger.info(f"{self.name} defined as a dask element")
                    return res
                logger.info(f"{self.name} computed, now saving")
                self.result_path.parent.mkdir(exist_ok=True, parents=True)
                pickle.dump(res, self.result_path.open("wb"))
                logger.info(f"{self.name} saved")
            else:
                logger.info(f"Loading {self.name}")
                res= pickle.load(self.result_path.open("rb"))
            logger.info(f"{self.name} is\n{res}")
            return res
        return new_f
    

@Autosave("counts", version = 6)
def get_counts():
    spectrogram_sig_count = spectrogram.aggregate_by(["t", "sig_num"])["sig_num"].apply(lambda d: d.nunique()).to_frame().compute().rename(columns={"sig_num":"sig_count"}).apply_index()
    spectrogram_window_count = spectrogram.aggregate_by(["t", "sig_num"])["amp"].count().to_frame().compute().rename(columns={"amp":"window_count"}).apply_index()
    return merge(spectrogram_sig_count, spectrogram_window_count)

@Autosave("pwelch", version = 3)
def get_pwelch():
    pwelch_window_median: pd.DataFrame = spectrogram.aggregate_by(["sig_num", "t"])["amp"].median().to_frame().compute().assign(agg_type = "median(t, sig_num)").apply_index()
    pwelch_window_avg: pd.DataFrame = spectrogram.aggregate_by(["sig_num", "t"])["amp"].mean().to_frame().compute().assign(agg_type = "mean(t, sig_num)").apply_index()
    
    pwelch_sig_median: pd.DataFrame = spectrogram.aggregate_by(["t"])["amp"].median().to_frame().reset_index().aggregate_by(["signal_num"])["amp"].median().to_frame().compute().assign(agg_type = "median(sig_num, median(t))").apply_index()
    pwelch_sig_mean: pd.DataFrame = spectrogram.aggregate_by(["t"])["amp"].mean().to_frame().reset_index().aggregate_by(["signal_num"])["amp"].mean().to_frame().compute().assign(agg_type = "mean(sig_num, mean(t))").apply_index()

    pwelch = pd.concat([pwelch_window_median, pwelch_window_avg, pwelch_sig_median, pwelch_sig_mean]).reset_index()
    return pwelch


def plot_pwelch(pwelch, counts):
    pwelch = merge(pwelch, counts).reset_index()
    pwelch=pwelch.loc[pwelch["freq"] < 45].copy()
    # pwelch["sig_count"] = pwelch["sig_count"].apply(lambda x: np.round(np.log10(x), 1))
    pwelch["window_count"] = pwelch["window_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    pwelch["Condition"] = pwelch["Healthy"].apply(lambda x: "Control" if x else "Park")
    hue_order = ["Rat", "Monkey", "Human"]
    window_size_order = pwelch["window_count"].drop_duplicates().sort_values(ascending=False).to_list()
    sig_sizes = {x:np.round(np.log10(x), 1) for x in pwelch["sig_count"].drop_duplicates()}
    pwelch_figs = toolbox.FigurePlot(data= pwelch, figures="sig_type", fig_title="Pwelch sig_type={sig_type}", col="agg_type", row="Structure", aspect=2, margin_titles=True)
    pwelch_figs.map(sns.scatterplot, x="freq", y="amp",  hue="Species", size="window_count", hue_order = hue_order, size_order = window_size_order, edgecolor="black", legend="full")
    pwelch_figs.map(sns.lineplot, x="freq", y="amp",  hue="Species", style="Condition", dashes=[(1, 2), (1, 0)], hue_order = hue_order, size="sig_count", sizes=sig_sizes)
    pwelch_figs.tight_layout().add_legend()

def plot_kde(kde):
    # pwelch = merge(pwelch, counts).reset_index()
    # pwelch=pwelch.loc[pwelch["freq"] < 45].copy()
    # # pwelch["sig_count"] = pwelch["sig_count"].apply(lambda x: np.round(np.log10(x), 1))
    # pwelch["window_count"] = pwelch["window_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    # pwelch["Condition"] = pwelch["Healthy"].apply(lambda x: "Control" if x else "Park")
    hue_order = ["Rat", "Monkey", "Human"]
    # window_size_order = pwelch["window_count"].drop_duplicates().sort_values(ascending=False).to_list()
    # sig_sizes = {x:np.round(np.log10(x), 1) for x in pwelch["sig_count"].drop_duplicates()}
    figs = toolbox.FigurePlot(data= kde, figures="sig_type", fig_title="Kernel Density Estimate, sig_type={sig_type}", row="Structure", aspect=2, margin_titles=True)
    # pwelch_figs.map(sns.scatterplot, x="amp", y="amp",  hue="Species", size="window_count", hue_order = hue_order, size_order = window_size_order, edgecolor="black", legend="full")
    figs.map(sns.lineplot, x="amp", y="density",  hue="Species", style="Condition", dashes=[(1, 2), (1, 0)], hue_order = hue_order)
    figs.tight_layout().add_legend()

@Autosave("band_spectrogram", version = 3)
def get_band_df():
    def mk_band(d):
        # logger.info(f"{type(d)}\n{d.to_string()}")
        # raise Exception("Stop")
        d = d.reset_index(drop=True).merge(bands_df, how="cross")
        d = d[(d["freq"] >= d["band_start"]) & (d["freq"] <= d["band_end"])]
        # logger.info(d)
        res = d.aggregate_by("freq").apply(lambda d: d["amp"].median() if d["band_agg"].iat[0] == "median" else 3)
        res = pd.DataFrame({"amp":res}).reset_index(level=["t", "band_name"]).reset_index(drop=True)
        # logger.info(f"\n{res.to_string()}")
        # raise Exception("Stop")
        return res
    return spectrogram.aggregate_by(["freq", "t"]).apply(mk_band, meta={"t":float, "band_name": str, "amp":float})

@Autosave("band_kde", version = 1)
def get_kde(bands):
    def compute_kde(d):
        print(d)
        input(d.values.shape)
        import scipy
        r = scipy.stats.gaussian_kde(d.values)
        (np.linspace(0, 1, 200))
        res = pd.DataFrame()
        res["amp"] = np.linspace(d.min(), d.max(), 200)
        res["density"] = r(res["amp"].values)
        return res
    
    kde = bands.aggregate_by(["sig_num", "t"])["amp"].apply(compute_kde, meta={"amp": float, "density": float}).compute()
    return kde

counts: pd.DataFrame = get_counts()
pwelch: pd.DataFrame = get_pwelch()
bands: dd.DataFrame  = get_band_df()
kde: pd.DataFrame = get_kde(bands)




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

