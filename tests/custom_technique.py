import pandas as pd, numpy as np, functools
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave

logger = logging.getLogger(__name__)
beautifullogger.setup()

nwindows,nsignals = 500, 50
nwindows,nsignals = None, None
bands = pd.DataFrame([[8, 15], [16, 30], [8, 30], [31, 49]], columns=["band_start","band_end"])


all_index_cols=["Species", "Structure","CorticalState", "Condition", "sig_type",  "sig_num", "t", "freq", "agg_type", "band_name", "kde_bw", "band_agg", "spectral_power"]
Autosave.nwindows,Autosave.nsignals, Autosave.result_base_path = nwindows, nsignals, "./ResultsCustom"
Autosave.param_name = f"{nwindows if not nwindows is None else 'all'},{nsignals if not nsignals is None else 'all'}"


@Autosave("counts", version = 0)
def get_counts(spectrogram):
    func = lambda df: pd.Series(dict(sig_count=len(df.columns), window_count=df.count().sum()))
    return spectrogram["newpath"].progress_apply(lambda path: func(pd.read_parquet(path)))

@Autosave("pwelch", version = 1)
def get_pwelch(spectrogram):
    def compute(df):
        return pd.Series({
            "median(t, sig_num)": df.median(axis=None),
            "mean(t, sig_num)": df.mean(axis=None), 
            "median(sig_num, median(t))": df.median(axis=1).median(axis=0), 
            "mean(sig_num, mean(t))": df.mean(axis=1).mean(axis=0), 
        }, name="pwelch_agg")

        
    pwelch = spectrogram["newpath"].progress_apply(lambda path: compute(pd.read_parquet(path)))
    pwelch.columns.name="agg_type"
    pwelch=pwelch.stack("agg_type")
    pwelch.name="spectral_power"
    return pwelch

@Autosave("band_spectrogram", version = 1)
def get_band_spectrogram(spectrogram):
    def file_func(paths, band_name):
        nsigs = len(pd.read_parquet(paths[0]).columns) #Assumes all have same number of signals
        series = {"median":{}, "mean":{}}
        for i in range(nsigs):
            s = pd.DataFrame([pd.read_parquet(path, columns=[str(i)]).squeeze() for path in paths])
            series["median"][i] = s.median()
            series["mean"][i] = s.mean()
        res = {agg: pd.DataFrame(se) for agg, se in series.items()}
        newpaths={}
        for agg,r in res.items():
            r.columns.name="sig_num"
            newpath = pathlib.Path("./BandSpectrogramData")/pathlib.Path(paths[0]).parent.relative_to("./DataParquetCustom")/f"band_data_{band_name},agg={agg}.parquet"
            newpath.parent.mkdir(exist_ok=True, parents=True)
            r.to_parquet(newpath)
            newpaths[agg] = str(newpath)
        return pd.Series(newpaths, name="path")

    def outer_func(d):
        return file_func(d.loc[(d["freq"] >= d["band_start"]) & (d["freq"] <=d["band_end"]), "newpath"].to_list(), d["band_name"].iat[0])
        # dfs = [pd.read_parquet(path) for path in paths]

    bandgram = spectrogram.reset_index().merge(bands, how="cross")
    bandgram= bandgram.aggregate_by(["freq"], group_keys=True).progress_apply(outer_func) 
    bandgram.columns.name = "band_agg"
    bandgram=bandgram.stack("band_agg")
    bandgram.name="path"
    return bandgram
    # return spectrogram.reset_index("freq").aggregate_by(["freq"]).apply(lambda d: input(d["freq"]))

@Autosave("band_kde", version = 1)
def get_kde(band_spectrogram: pd.DataFrame):
    def compute_kde(path):
        data = pd.read_parquet(path).values.reshape(-1)
        data=data[~np.isnan(data)]
        import scipy
        kdes = {bw:scipy.stats.gaussian_kde(data, bw_method=bw/data.std()) for bw in [10**(-i) for i in range(2, 6)]}
        res = pd.DataFrame({"kde":pd.Series(kdes)})
        res.index.name="kde_bw"
        res = res.apply(lambda row: pd.Series(row["kde"](np.linspace(0, 0.010, 200)), index=np.linspace(0, 0.010, 200), name="density"), axis=1)
        res.columns.name = "spectral_power"
        res = res.stack("spectral_power")
        res.name="density"
        return res
    
    kde = band_spectrogram.progress_apply(compute_kde)
    kde = kde.stack(["kde_bw", "spectral_power"])
    kde.name="density"
    return kde

@Autosave("band_distribution", version = 5, debug=False)
def get_band_distribution(band_spectrogram: pd.DataFrame):
    def compute(path, amps):
        df = pd.read_parquet(path)
        all_values = np.sort(df.values.reshape(-1))
        all_values = all_values[~np.isnan(all_values)]
        res = np.searchsorted(all_values, amps)/all_values.size
        return pd.Series(res, index=amps, name="quantile")
    
    quantiles = band_spectrogram.progress_apply(compute, amps=np.linspace(0.0001, 0.01, 2000))
    quantiles.columns.name = "spectral_power"
    quantiles = quantiles.stack("spectral_power")
    quantiles.name="quantile"
    return quantiles


@Autosave("thresholds", version = 3, debug=False)
def get_thresholds(band_distribution: pd.DataFrame):
    df = band_distribution.unstack("Condition")
    df["Diff"] = np.abs(df["Control"] - df["Park"])
    df = df.sort_values("Diff", ascending=False)
    df = (
            df.reset_index()
            .drop_duplicates(subset=[col for col in all_index_cols if (col in df.columns or col in df.index.names) and not col in ["spectral_power"]])
            .rename(columns={
                "spectral_power": "threshold_spectral_power", 
                "Diff": "threshold_difference", 
                # "band_name": "threshold_band", 
                "Control": "quantile_ctrl", "Park": "quantile_park"
            })
            .apply_index(append=False)
    )
    df["Best_threshold"] = df.aggregate_by(["band_name"])["threshold_difference"].transform(lambda d: d.max())
    df["is_best"] = df["threshold_difference"] == df["Best_threshold"]
    df = df.drop(columns=["Best_threshold"])
    return df

def main(spectrogram):
    tqdm.tqdm.pandas(desc="Computing")
    counts = get_counts(spectrogram)
    pwelch = get_pwelch(spectrogram)
    band_spectrogram= get_band_spectrogram(spectrogram)
    kde = get_kde(band_spectrogram)
    band_distribution = get_band_distribution(band_spectrogram)
    thresholds = get_thresholds(band_distribution)
    logger.info("Plotting")
    plot_pwelch(pwelch, counts)
    plot_kde(kde)
    plot_band_distribution(band_distribution, thresholds)
    plt.show()



def plot_pwelch(pwelch, counts):
    pwelch = merge(pwelch, counts).reset_index()
    pwelch=pwelch.loc[pwelch["freq"] < 45].copy()
    pwelch["window_count"] = pwelch["window_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    pwelch["sig_count"] = pwelch["sig_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    window_size_order = pwelch["window_count"].drop_duplicates().sort_values(ascending=False).to_list()
    sig_sizes = pwelch["sig_count"].drop_duplicates().sort_values(ascending=False).to_list()

    pwelch_figs = toolbox.FigurePlot(data=pwelch, 
        figures="sig_type", fig_title="Pwelch sig_type={sig_type}", 
        col="agg_type", row="Structure", row_order=structure_order, 
        aspect=2, margin_titles=True
    )
    # pwelch_figs.map(sns.scatterplot, x="freq", y="spectral_power",  
    #     hue="Species", hue_order = species_order, 
    #     size="window_count", size_order = window_size_order, 
    #     edgecolor="black"
    # )
    pwelch_figs.map(sns.lineplot, x="freq", y="spectral_power",  
        hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
        size="sig_count", size_order=sig_sizes
    )
    pwelch_figs.tight_layout().add_legend()

def plot_kde(kde: pd.DataFrame):
    kde = kde.xs(0.00001, level="kde_bw", drop_level=False)
    figs = toolbox.FigurePlot(data=kde, 
        figures=["sig_type", "band_agg", "kde_bw"], 
        fig_title="Gaussian Kernel Density Estimate, sig_type={sig_type}, band_agg={band_agg}, kde_bandwith={kde_bw}", 
        row="Structure", row_order=structure_order,
        col="band_name",  col_order=bands["band_name"].drop_duplicates(),
        aspect=2, margin_titles=True
    )
    figs.map(sns.lineplot, x="spectral_power", y="density",  
        hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
    )
    figs.tight_layout().add_legend()

def plot_band_distribution(band_distribution, thresholds):
    band_distribution = merge(band_distribution, thresholds)
    figs = toolbox.FigurePlot(data=band_distribution, 
        figures=["sig_type", "band_agg"], 
        fig_title="Distribution in bands, sig_type={sig_type}, band_agg={band_agg}", 
        row="Structure", row_order=structure_order,
        col="Species",  col_order=species_order,
        aspect=2, margin_titles=True
    )
    
    def my_vlines(x, ymin, ymax, *args, hue, hue_order, color, data: pd.DataFrame, **kwargs):
        import matplotlib.patheffects as mpe
        data = data.drop_duplicates(subset=[x, ymin, ymax, hue])
        data_best = data.loc[data["is_best"]]
        data_other = data.loc[~data["is_best"]]
        plt.vlines(x, ymin, ymax, *args, **kwargs, data=data_other, color=[f"C{hue_order.index(h)}" for h in data_other[hue]])
        plt.vlines(x, ymin, ymax, *args, **kwargs, data=data_best, color=[f"C{hue_order.index(h)}" for h in data_best[hue]], path_effects=[mpe.withStroke(linewidth=3, foreground='black')])

    figs.map(my_vlines, x="threshold_spectral_power", ymin="quantile_ctrl", ymax="quantile_park",
             hue="band_name", hue_order = bands["band_name"].drop_duplicates().to_list()
    )
    figs.map(sns.lineplot, x="spectral_power", y="quantile",  
        hue="band_name", hue_order = bands["band_name"].drop_duplicates(),
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
    )
    figs.tight_layout().add_legend()






def aggregate_by(self, by, *args, **kwargs):
    global all_index_cols
    keys = [col for col in all_index_cols if not col in by and (col in self.columns or col in self.index.names)]
    return self.groupby(keys, *args, **kwargs, observed=True)

pd.DataFrame.aggregate_by = aggregate_by
pd.Series.aggregate_by = aggregate_by

def apply_index(self: pd.DataFrame, append=True):
    return self.set_index([col for col in all_index_cols if col in self.columns], append=append)
pd.DataFrame.apply_index = apply_index

def new_str(self):
    old = self.old_str()
    addition = f"Index names = {self.index.names}\nColumns = {self.columns if hasattr(self, 'columns') else self.name}"
    return f"{old}\n{addition}"
pd.DataFrame.old_str = pd.DataFrame.__str__
pd.DataFrame.__str__ = new_str
pd.Series.old_str = pd.Series.__str__
pd.Series.__str__ = new_str

def merge(a, b):
    a=a.reset_index()
    b=b.reset_index()
    on_a = set(a.index.names).union(set(a.columns)).intersection(set(all_index_cols))
    on_b = set(b.index.names).union(set(b.columns)).intersection(set(all_index_cols))
    on = list(on_a.intersection(on_b))
    # print(a)
    # print(b)
    # input(on)
    return a.reset_index().merge(b.reset_index(), how="outer", on=on).apply_index(append=False)

class PathTransformer: pass
class PathInput: pass

species_order = ["Rat", "Monkey", "Human"]
structure_order=["STN", "GPe", "STR"]
sig_type_order=["lfp", "bua", "spikes"]
condition_order = ["Park", "Control"]

if __name__ =="__main__":
    sns.set_theme()
    spectrogram = pd.read_parquet(pathlib.Path("." ) / "DataParquetCustom"/ f"{Autosave.param_name}.parquet")
    bands["band_name"] = bands.apply(lambda row: f"{row['band_start']}, {row['band_end']}", axis=1)
    bands.set_index("band_name")
    main(spectrogram.apply_index())

