import pandas as pd, numpy as np, functools, scipy
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


all_index_cols=["Species", "Structure","CorticalState", "Condition", "sig_type",  "sig_num", "t", "freq", "agg_type", "band_name", "kde_bw", "band_agg", "spectral_power", "oscilating", "%t-burst", "burst_id"]
Autosave.nwindows,Autosave.nsignals, Autosave.result_base_path = nwindows, nsignals, "./ResultsCustom"
Autosave.param_name = f"{nwindows if not nwindows is None else 'all'},{nsignals if not nsignals is None else 'all'}"


@Autosave("counts", version = 1)
def get_counts(spectrogram):
    func = lambda df: pd.Series(dict(sig_count=len(df.columns), window_count=df.count().sum()))
    return spectrogram["newpath"].progress_apply(lambda path: func(pd.read_parquet(path)))

@Autosave("pwelch", version = 8)
def get_pwelch(spectrogram):
    def compute(df: pd.DataFrame):
        vals = remove_na(df.values.reshape(-1))
        res = pd.Series({
            # "q(0.75, (t, sig_num))": np.percentile(remove_na(df.values.reshape(-1)), 75),
            "median(t, sig_num)": np.median(vals),
            "mean(t, sig_num)": np.mean(vals), 
            # "median(sig_num, median(t))": df.median(axis=1).median(axis=0), 
            # "mean(sig_num, mean(t))": df.mean(axis=1).mean(axis=0), 
            "cv(t, sig_num)": scipy.stats.variation(vals)
        }, name="pwelch_agg")
        return res

        
    pwelch = spectrogram["newpath"].progress_apply(lambda path: compute(pd.read_parquet(path)))
    pwelch.columns.name="agg_type"
    pwelch=pwelch.stack("agg_type")
    pwelch.name="spectral_power"
    return pwelch


@Autosave("band_spectrogram", version = 2)
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

@Autosave("band_kde", version = 3)
def get_kde(band_spectrogram: pd.DataFrame):
    def compute_kde(path):
        data = pd.read_parquet(path).values.reshape(-1)
        data=data[~np.isnan(data)]
        data=data[data > 5*10**(-5)]
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

@Autosave("band_dist_fit", version = 2, debug=False)
def get_band_dist_fit(band_spectrogram: pd.DataFrame):
    def compute(path):
        data = pd.read_parquet(path).values.reshape(-1)
        data=data[~np.isnan(data)]
        data=data[data > 5*10**(-5)]
        import scipy
        fitted = scipy.stats.skewnorm.fit(data, loc=0.001)
        return fitted
        # kdes = {bw:scipy.stats.gaussian_kde(data, bw_method=bw/data.std()) for bw in [10**(-i) for i in range(2, 6)]}
        # res = pd.DataFrame({"kde":pd.Series(kdes)})
        # res.index.name="kde_bw"
        # res = res.apply(lambda row: pd.Series(row["kde"](np.linspace(0, 0.010, 200)), index=np.linspace(0, 0.010, 200), name="density"), axis=1)
        # res.columns.name = "spectral_power"
        # res = res.stack("spectral_power")
        # res.name="density"
        return res
    
    res = band_spectrogram.sel(band_agg="mean").progress_apply(compute)
    res_plot = res.progress_apply(lambda x: pd.Series(scipy.stats.skewnorm.pdf(np.linspace(0, 0.010, 200), *x),  index=np.linspace(0, 0.010, 200)))
    res_plot.columns.name = "spectral_power"
    res_plot = res_plot.stack("spectral_power")
    res_plot.name="density"
    return res, res_plot

@Autosave("band_distribution", version = 6, debug=False)
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


@Autosave("thresholds", version = 4, debug=False)
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
                "Control": "quantile_ctrl", "Park": "quantile_park"
            })
            .apply_index(append=False)
    )
    df["Best_threshold"] = df.aggregate_by(["band_name"])["threshold_difference"].transform(lambda d: d.max())
    df["is_best"] = df["threshold_difference"] == df["Best_threshold"]
    df = df.drop(columns=["Best_threshold"])
    return df



@Autosave("burst_properties", version = 4, debug=False)
def get_burst_properties(band_spectrogram: pd.DataFrame, thresholds: pd.DataFrame):
    def get_properties(signal: pd.Series, power):
        d = pd.DataFrame()
        d["signal"] = signal.T.dropna()
        d["oscilating"] = d["signal"] >=power
        d["burst_id"] = d["oscilating"].diff().ne(0).cumsum()
        
        #Place to compute properties for each signal
        res = d.reset_index("t").groupby("burst_id").agg(
                oscilating= pd.NamedAgg("oscilating", "min"), 
                band_spectral_max_power= pd.NamedAgg("signal", "max"), 
                band_spectral_mean_power= pd.NamedAgg("signal", "mean"), 
                band_spectral_median_power= pd.NamedAgg("signal", "median"), 
                start= pd.NamedAgg("t", "min"), 
                end= pd.NamedAgg("t", "max")
        )

        res["end"] = res["end"] + 0.05
        res["Duration"] = res["end"] - res["start"]
        res["sig_avg_amp"] = d["signal"].mean()
        return res
    def compute(path, power):
        d = pd.read_parquet(path).T
        res: pd.DataFrame = d.groupby(["sig_num"]).apply(get_properties, power=power)
        oscilating = res.loc[res["oscilating"]]
        newpath_oscilating = pathlib.Path("./BandBurstMetaData")/pathlib.Path(path).parent.relative_to("./BandSpectrogramData")/pathlib.Path(path).stem/ f"burst_data_oscilating.parquet"
        newpath_oscilating.parent.mkdir(exist_ok=True, parents=True)
        oscilating.to_parquet(newpath_oscilating)

        non_oscilating = res.loc[~res["oscilating"]]
        newpath_nonoscilating = pathlib.Path("./BandBurstMetaData")/pathlib.Path(path).parent.relative_to("./BandSpectrogramData")/pathlib.Path(path).stem/ f"burst_data_nonoscilating.parquet"
        newpath_nonoscilating.parent.mkdir(exist_ok=True, parents=True)
        non_oscilating.to_parquet(newpath_nonoscilating)
        return pd.Series({True: str(newpath_oscilating), False:str(newpath_nonoscilating)})
    
    df: pd.DataFrame = merge(band_spectrogram, thresholds.loc[thresholds["is_best"], "threshold_spectral_power"].reset_index(level="band_name").drop(columns="band_name"))
    df=df[~df["threshold_spectral_power"].isna()]
    df = df.aggregate_by([]).progress_apply(lambda row: compute(row["path"].iat[0], row["threshold_spectral_power"].iat[0]))
    df.columns.name = "oscilating"
    df = df.stack("oscilating")
    df.name="path"
    return df

@Autosave("burst_group_info", version = 2, debug=False)
def get_burst_group_info(burst_properties: pd.DataFrame):
    def compute(path):
        d = pd.read_parquet(path)
        res = pd.Series({
            "duration: mean(sig, mean(burst))": (d.groupby("sig_num")["Duration"].mean()).mean(),
            "duration: mean(sig, burst)": d["Duration"].mean(),
            "mean_spectral_power: mean(sig, burst)": d["band_spectral_mean_power"].mean(),
            "mean_spectral_power: mean(sig, mean(burst))":  (d.groupby("sig_num")["band_spectral_mean_power"].mean()).mean(),
            "n_burst/s: mean(sig)": (d.groupby("sig_num").apply(lambda s: s["band_spectral_mean_power"].count()/s["end"].max())).mean(),
            "sig_val: mean(sig, mean(t))": (d.groupby("sig_num")["sig_avg_amp"].head(1)).mean()
        })
        return res
    
    return burst_properties.progress_apply(compute)

@Autosave("burst_sig_info", version = 2, debug=False)
def get_burst_sig_info(burst_properties: pd.DataFrame):
    def compute(path):
        path = path.iat[0]
        d = pd.read_parquet(path)
        res = pd.DataFrame()
        res["avg_duration"] = d.groupby("sig_num")["Duration"].mean()
        # input(res)
        r=  d.groupby("sig_num")["sig_avg_amp"].head(1).reset_index("burst_id", drop=True)
        # input(r)
        res["avg_sig_power"] = r
        # input(res)
        return res
    
    return burst_properties.reset_index().aggregate_by([])["path"].progress_apply(compute)




@Autosave("burst_spectrogram", version = 1, debug=False)
def get_burst_spectrogram(spectrogram: pd.DataFrame, burst_properties: pd.DataFrame):
    def get_data(spec, sig, start, end):
        start= np.round(start, 5)
        end = np.round(end, 5)
        # print(start, end)
        
        s = spec.loc[start:end, sig].iloc[:-1]
        m, M = s.index[0], s.index[-1]
        
        # if m == M:
        #     return None
        
        new_coords = np.linspace(0, 1, 201)
        res = pd.Series(np.interp(new_coords*(M-m)+m, s.index.values, s.values), index=new_coords, name="spectral_power")  
        res.index.name = "%t-burst"
        return res
    def compute(spectrogram_path, burst_path):
        # with toolbox.Profile() as p:
        spec = pd.read_parquet(spectrogram_path)
        burst = pd.read_parquet(burst_path).reset_index()
        spec = spec.stack()
        spec.name = "spectral_power_"
        

        sqlcode = '''
            SELECT 
                burst.sig_num,
                burst.burst_id,
                spec.spectral_power_,
                spec.t,
                burst.start,
                burst.end
            from burst
            INNER JOIN spec 
                on spec.sig_num=burst.sig_num and spec.t >= burst.start and spec.t < burst.end 
                
        '''

        import pandasql as ps
        # conn = sqlite3.connect(':memory:')
        #write the tables
        # burst.to_sql('burst', conn, index=False)
        # spec.reset_index().to_sql('spec', conn, index=False)
        # and spec.t between burst.start and burst.end    spec.sig_num=burst.sig_num 
        # r = pd.read_sql_query(sqlcode, conn)
        # input(spec.reset_index())
        r = ps.sqldf(sqlcode,dict(burst=burst, spec=spec.reset_index()))


        r=r.set_index(["sig_num", "burst_id", "t"])
        def interpolate(t):
            t = t["spectral_power_"].values
            new_coords = np.linspace(0, 1, 101)
            return pd.Series(np.interp(new_coords, np.linspace(0, 1, t.size), t), index=new_coords, name="%t-burst")  

        r = r.aggregate_by("t").apply(interpolate)
        
        # r = merge(burst, spec)
        # input(r)
        # def test(row):
        #     r = get_data(spec, row["sig_num"], row["start"], row["end"])
        #     r = pd.concat({row["sig_num"]: r}, names=["sig_num"])
        #     return r
        # res = burst.reset_index("sig_num").progress_apply(test, axis=1)
        # res.columns.name = "%t-burst"
        # print(res)
        path = pathlib.Path("./BurstSpectrogram")/pathlib.Path(burst_path).parent.relative_to("./BandBurstMetaData")/pathlib.Path(burst_path).stem/ f"{pathlib.Path(spectrogram_path).stem}.parquet"
        path.parent.mkdir(exist_ok=True, parents=True)
        r.to_parquet(path)
        
        # pr = p.get_results()
        # # print(pr)
        # # print(pr["filename"].str.contains("custom_technique", regex=False, na=False))
        # pr = pr.loc[pr["filename"].astype(str).str.contains("custom", regex=False, na=False), :]
        # print(pr.sort_values("tottime", ascending=False).head(30))
        
        return str(path)

    r = merge(spectrogram, burst_properties)
    r = r[~r["path"].isna()]
    res = r[["newpath", "path"]].progress_apply(lambda row: compute(row["newpath"], row["path"]), axis=1)
    return res

def dist_examples(df: pd.DataFrame):
    df = df.sel(Species="Rat", Structure="GPe", band_name="8, 30", band_agg="mean")
    def compute(path):
        path = path.iat[0]
        d = pd.read_parquet(path).T.sample(12)

        d = d.diff(axis=1, periods=10)

        def get_kde(vals, bw):
            data=remove_na(vals.values)
            return scipy.stats.gaussian_kde(data, bw_method=bw/data.std())
        d = d.apply(lambda row: get_kde(row, 10**(-5)), axis=1).reset_index()
        d["sig_num_bis"] = [i for i in range(len(d.index))]
        d=d.set_index(["sig_num", "sig_num_bis"])
        d = d[0]
        # res = d.apply(lambda kde: pd.Series(kde(np.linspace(0.0001, 0.020, 201)), index=np.linspace(0.0001, 0.020, 201), name="density"))
        res = d.apply(lambda kde: pd.Series(kde(np.linspace(-0.020, 0.020, 1000)), index=np.linspace(-0.020, 0.020, 1000), name="density"))
        
        res.columns.name="spectral_power"
        res = res.stack()
        res.name="density"
        return res
    df = df.reset_index().aggregate_by([])["path"].progress_apply(compute)
    df.name="density"
    # input(df)
    mpl.rcParams["axes.titlesize"] = 8
    figs = toolbox.FigurePlot(data=df, 
        figures=["sig_type", "Species", "Structure", "band_name", "band_agg"], fig_title="Examples Species={Species}, Structure={Structure}, sig_type={sig_type}, band={band_name}, band_agg={band_agg}", 
        col="sig_num_bis", subplot_title="sig={sig_num}", col_wrap=3,
        aspect=2, margin_titles=True
    )

    figs.map(sns.lineplot, y = "density", x="spectral_power", hue="Condition", hue_order=condition_order)
    # figs.map(plt.hlines, y="threshold_spectral_power", xmin=0, xmax=df.reset_index("t")["t"].max(), color="green", linewidth=1)
    # figs.map(sns.lineplot, y = "oscilating_display", x="t", color="red")
    # figs.map(sns.scatterplot, y = "signal", x="t", hue="oscilating", hue_order=[True, False])
    
    figs.tight_layout().add_legend()
        
def main(spectrogram):
    tqdm.tqdm.pandas(desc="Computing")
    counts = get_counts(spectrogram)
    pwelch = get_pwelch(spectrogram)
    # band_spectrogram= get_band_spectrogram(spectrogram)
    # kde = get_kde(band_spectrogram)
    # dist_fit, dist_fit_plot = get_band_dist_fit(band_spectrogram)
    # band_distribution = get_band_distribution(band_spectrogram)
    # thresholds = get_thresholds(band_distribution)
    
    # burst_properties = get_burst_properties(band_spectrogram, thresholds)
    # burst_group_info = get_burst_group_info(burst_properties)
    # burst_sig_info = get_burst_sig_info(burst_properties)
    # dist_ex = dist_examples(band_spectrogram)
    # burst_spectrogram = get_burst_spectrogram(spectrogram, burst_properties)
    logger.info("Plotting")
    plot_pwelch(pwelch, counts)
    # plot_kde(kde, dist_fit_plot)
    # plot_band_distribution(band_distribution, thresholds)
    # plot_sig_examples(band_spectrogram, thresholds)
    # plot_group_info(burst_group_info)
    # plot_burst_sig_info(burst_sig_info)
    plt.show()

def plot_group_info(burst_group_info: pd.DataFrame):
    # mpl.rcParams["axes.titlesize"] = 8
    burst_group_info = burst_group_info.reset_index().copy()
    burst_group_info["n_burst/s: avg(sig)"] = burst_group_info["n_burst/s: mean(sig)"].apply(lambda x: f"~{10 ** np.round(np.log10(x), 1):.2g}")
    burst_group_info["mean_power\nmean(sig, burst)"] = burst_group_info["mean_spectral_power: mean(sig, burst)"]
    # burst_group_info["combined"] = burst_group_info[["band_name"]].apply(lambda row: ", ".join([f"{k}={v}" for k,v in row.items()]), axis=1)
    figs = toolbox.FigurePlot(data=burst_group_info, 
        figures=["sig_type", "band_agg", "oscilating"], fig_title="Burst Data sig_type={sig_type}, band_agg={band_agg}, oscilating={oscilating}", 
        row="Structure" , col="band_name",
        aspect=2, margin_titles=True
    )
    figs.map(sns.scatterplot, 
        x="duration: mean(sig, burst)", y="mean_power\nmean(sig, burst)", 
        size="n_burst/s: avg(sig)", size_order=burst_group_info["n_burst/s: avg(sig)"].drop_duplicates().sort_values(ascending=False).to_list(),
        style="Condition", style_order=condition_order,
        hue="Species", hue_order=species_order
    )
    figs.tight_layout().add_legend()

def plot_burst_sig_info(burst_sig_info: pd.DataFrame):
    # input(burst_sig_info)
    burst_sig_info = burst_sig_info.xs("mean", level="band_agg", drop_level=False).xs(True, level="oscilating", drop_level=False).xs("8, 30", level="band_name", drop_level=False)
    # mpl.rcParams["axes.titlesize"] = 8
    # burst_group_info = burst_group_info.reset_index().copy()
    # burst_group_info["n_burst/s: avg(sig)"] = burst_group_info["n_burst/s: mean(sig)"].apply(lambda x: f"~{10 ** np.round(np.log10(x), 1):.2g}")
    # burst_group_info["mean_power\nmean(sig, burst)"] = burst_group_info["mean_spectral_power: mean(sig, burst)"]
    # burst_group_info["combined"] = burst_group_info[["band_name"]].apply(lambda row: ", ".join([f"{k}={v}" for k,v in row.items()]), axis=1)
    figs = toolbox.FigurePlot(data=burst_sig_info, 
        figures=["sig_type", "band_agg", "oscilating", "band_name"], fig_title="Burst dur/vs avgamp sig_type={sig_type}, band_name = {band_name}, band_agg={band_agg}, oscilating={oscilating}", 
        row="Structure" , col="Species",
        aspect=2, margin_titles=True,
    )

    figs.map(sns.scatterplot, 
        y="avg_sig_power", x="avg_duration", 
        # style="Condition", style_order=condition_order,
        hue="Condition", hue_order=condition_order
    )
    def custom_regplot(x, y, *args, hue, hue_order, color, data: pd.DataFrame, **kwargs):
        groups = data.groupby(hue, observed=True)
        for h, d in groups:
            sns.regplot(data=d, x=x, y=y, color=f"C{hue_order.index(h)}", *args, **kwargs)

    figs.map(custom_regplot, y="avg_sig_power", x="avg_duration", scatter=False, logx=True, hue="Condition", hue_order=condition_order)
    # figs.map(custom_regplot, y="avg_sig_power", x="avg_duration", scatter=False, order=4, hue="Condition", hue_order=condition_order, line_kws=dict(linestyle=":"))
    figs.tight_layout().add_legend()

def plot_pwelch(pwelch, counts):
    
    pwelch=pwelch[pwelch.index.get_level_values("agg_type").isin(["mean(t, sig_num)"])]
    pwelch = merge(pwelch, counts).reset_index()
    pwelch=pwelch.loc[pwelch["freq"] < 45].copy()
    pwelch["window_count"] = pwelch["window_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    pwelch["sig_count"] = pwelch["sig_count"].apply(lambda x: f"~{10 ** np.round(np.log10(x))}")
    window_size_order = pwelch["window_count"].drop_duplicates().sort_values(ascending=False).to_list()
    sig_sizes = pwelch["sig_count"].drop_duplicates().sort_values(ascending=False).to_list()

    pwelch_figs = toolbox.FigurePlot(data=pwelch, 
        figures="sig_type", fig_title="Pwelch sig_type={sig_type}, {agg_type}", 
        col="Species", row="Structure", row_order=structure_order, col_order=species_order,
        aspect=2, margin_titles=True
    )
    # pwelch_figs.map(sns.scatterplot, x="freq", y="spectral_power",  
    #     hue="Species", hue_order = species_order, 
    #     size="window_count", size_order = window_size_order, 
    #     edgecolor="black"
    # )
    pwelch_figs.map(sns.lineplot, x="freq", y="spectral_power",  
        # hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
        # size="sig_count", size_order=sig_sizes
    )
    pwelch_figs.tight_layout().add_legend()

def plot_kde(kde: pd.DataFrame, dist_plot:pd.DataFrame):
    kde = kde.xs(0.00001, level="kde_bw", drop_level=False)
    dist_plot.name="fit_density"
    kde.name="kde_density"
    kde = merge(kde, dist_plot)
    figs = toolbox.FigurePlot(data=kde, 
        figures=["sig_type", "band_agg", "kde_bw"], 
        fig_title="Gaussian Kernel Density Estimate, sig_type={sig_type}, band_agg={band_agg}, kde_bandwith={kde_bw}", 
        row="Structure", row_order=structure_order,
        col="band_name",  col_order=bands["band_name"].drop_duplicates(),
        aspect=2, margin_titles=True
    )
    figs.map(sns.lineplot, x="spectral_power", y="kde_density",  
        hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order,
    )
    figs.map(sns.lineplot, x="spectral_power", y="fit_density",  
        hue="Species", hue_order = species_order,
        style="Condition", dashes=[(1, 0), (1, 2)], style_order=condition_order, alpha=0.5,
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



def plot_sig_examples(band_spectrogram:pd.DataFrame, thresholds: pd.DataFrame):
    def get_properties(signal: pd.Series, power, group_path):
        d = pd.DataFrame()
        d["signal"] = signal.T.dropna()
        d["oscilating"] = d["signal"] >=power
        d["burst_id"] = d["oscilating"].diff().ne(0).cumsum()
        d["oscilating_display"] = (d["oscilating"]) *power /3
        

        other_info = d.reset_index("t").groupby("burst_id").agg(
                oscilating= pd.NamedAgg("oscilating", "min"), 
                band_spectral_mean_power= pd.NamedAgg("signal", "mean"), 
                band_spectral_median_power= pd.NamedAgg("signal", "median"), 
                start= pd.NamedAgg("t", "min"), 
                end= pd.NamedAgg("t", "max")
        )
        other_info["end"] = other_info["end"] + 0.05
        d["avg_duration"] = (other_info["end"] - other_info["start"]).loc[other_info["oscilating"]].mean()
        d["avg_amp"] = (d["signal"]).mean()
        d["avg_on_burst_amp"] = (other_info.loc[other_info["oscilating"], "band_spectral_mean_power"]).mean()
        return d
    

    def compute(path, power):
        d = pd.read_parquet(path).T.sample(n=4)
        d["sig_num_bis"] = [i for i in range(len(d.index))]
        d = d.set_index("sig_num_bis", append=True)
        res: pd.DataFrame = d.groupby(["sig_num", "sig_num_bis"]).apply(get_properties, power=power, group_path=path)
        res["threshold_spectral_power"] = power
        return res
    
    df: pd.DataFrame = merge(band_spectrogram, thresholds.loc[thresholds["is_best"], "threshold_spectral_power"].reset_index(level="band_name").drop(columns="band_name"))
    df=df[~df["threshold_spectral_power"].isna()]
    df=df.xs("Monkey", level="Species", drop_level=False)
    df=df.xs("STN", level="Structure", drop_level=False)
    df=df.xs("8, 30", level="band_name", drop_level=False)
    df=df.xs("median", level="band_agg", drop_level=False)
    df = df.aggregate_by([]).progress_apply(lambda row: compute(row["path"].iat[0], row["threshold_spectral_power"].iat[0]))

    mpl.rcParams["axes.titlesize"] = 8
    figs = toolbox.FigurePlot(data=df, 
        figures=["sig_type", "Species", "Structure", "band_name", "band_agg"], fig_title="Examples Species={Species}, Structure={Structure}, sig_type={sig_type}, band={band_name}, band_agg={band_agg}", 
        col="Condition", row="sig_num_bis", col_order=["Park", "Control"], subplot_title="sig={sig_num}, avg_amp={avg_amp:.4g}, avg_burst_duration={avg_duration:.4g}, avg_on_burst_amp={avg_on_burst_amp:.4g}",
        aspect=2, margin_titles=True
    )

    figs.map(sns.lineplot, y = "signal", x="t")
    figs.map(plt.hlines, y="threshold_spectral_power", xmin=0, xmax=df.reset_index("t")["t"].max(), color="green", linewidth=1)
    figs.map(sns.lineplot, y = "oscilating_display", x="t", color="red")
    figs.map(sns.scatterplot, y = "signal", x="t", hue="oscilating", hue_order=[True, False])
    
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

def sel(self: pd.DataFrame, **kwargs):
    r = self
    for k,v in kwargs.items():
        r = r.xs(v, level=k, drop_level=False)
    return r

pd.DataFrame.sel = sel
pd.Series.sel=sel

def remove_na(self: np.ndarray):
    if not self.ndim==1:
        raise Exception("remove na only valid for 1d arrays")
    else:
        return self[~np.isnan(self)]
    

def merge(a, b):
    a=a.reset_index()
    b=b.reset_index()
    on_a = set(a.index.names).union(set(a.columns)).intersection(set(all_index_cols))
    on_b = set(b.index.names).union(set(b.columns)).intersection(set(all_index_cols))
    on = list(on_a.intersection(on_b))
    # print(a)
    # print(b)
    # input(on)
    return a.merge(b, how="outer", on=on).apply_index(append=False)




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

