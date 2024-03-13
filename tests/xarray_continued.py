import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave
from xarray_helper import apply_file_func, auto_remove_dim, nunique, apply_file_func_decorator, extract_unique

if not __name__=="__main__":
    exit()

xr.set_options(use_flox=True, display_expand_coords=True, display_max_rows=100, display_expand_data_vars=True, display_width=150)
logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
logging.getLogger("flox").setLevel(logging.WARNING)
tqdm.tqdm.pandas(desc="Computing")

MODE: Literal["TEST", "ALL", "SMALL", "BALANCED"]="ALL"
force_recompute = False
match MODE:
    case "TEST":
        cache_path = "/media/julien/data1/JulienCache/Test/"
        dtypes = dict(Contact=np.int32, Healthy=np.int8, Session=object)
        signals = (
            pd.read_csv(cache_path+"test_data.tsv", sep="\t", date_format="%d%M%Y", parse_dates=["Date"], dtype=dtypes)
            .set_index(["Contact", "sig_preprocessing"])
            .to_xarray().set_coords(["Session", "Structure", "FullStructure", "Date", "Condition", "Healthy", "CorticalState", "Species"])
        )
        signals = auto_remove_dim(signals, dim_list=["sig_preprocessing"])
        
        for c, d in dtypes.items():
            signals[c] = signals[c].astype(d)
        x=np.linspace(0, 20, 10001)
        test_signal=xr.DataArray(data=np.cos(10*x*2*np.pi)+np.cos(20*x*2*np.pi), coords=dict(t=x))
        for i in range(1, 6):
            pickle.dump(test_signal, open(f"./sig_{i}.pkl", "wb"))
        recompute=True
    case "ALL" | "BALANCED":
        cache_path = f"/media/julien/data1/JulienCache/All/"
        signals = xr.open_dataset(cache_path+"../signals.nc").load()
        recompute=False
    case "SMALL":
        cache_path = "/media/julien/data1/JulienCache/Small/"
        signals = xr.open_dataset(cache_path+"../signals.nc").load().head(dict(Contact="50"))
        recompute=False
    case "BALANCED":
        cache_path = f"/media/julien/data1/JulienCache/Balanced/"
        signals = xr.open_dataset(cache_path+"../signals.nc").load()
        group_index_cols = ["Species", "Structure", "Healthy"]
        signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
        signals = signals.set_coords("group_index")
        signals = signals.groupby("group_index").map(lambda x: x.head(20))
        signals = signals.drop("group_index")
recompute = recompute or force_recompute


signals["time_representation_path"] = xr.apply_ufunc(lambda x: np.where(x=="", np.nan, x), signals["time_representation_path"])
signals = signals.where(signals["Structure"].isin(["GPe", "STN", "STR"]), drop=True)

print(signals)
###### Now let's compute stuff on our signals
@apply_file_func_decorator(".", name="spike_duration", save_group=cache_path+"spike_durations.pkl", recompute=recompute)
def compute_spike_duration(arr: xr.DataArray):
    return float(np.max(arr) - np.min(arr))

@apply_file_func_decorator(".", name="sig_duration", save_group=cache_path+"sig_durations.pkl", recompute=recompute)
def compute_sig_duration(arr: xr.DataArray):
    return arr["t"].max() - arr["t"].min()

@apply_file_func_decorator(".", name="spike_start", save_group=cache_path+"spike_start.pkl", recompute=recompute)
def compute_spike_start(arr: xr.DataArray):
    return float(np.min(arr))

@apply_file_func_decorator(".", name="sig_start", save_group=cache_path+"sig_start.pkl", recompute=recompute)
def compute_sig_start(arr: xr.DataArray):
    return arr["t"].min()

print(signals)
signals["duration"] = xr.where(signals["sig_type"]=="spike_times", 
    compute_spike_duration(signals["time_representation_path"].where(signals["sig_type"] == "spike_times")),
    compute_sig_duration(signals["time_representation_path"].where(signals["sig_type"] != "spike_times"))
)

signals["start_time"] = xr.where(signals["sig_type"]=="spike_times", 
    compute_spike_start(signals["time_representation_path"].where(signals["sig_type"] == "spike_times")),
    compute_sig_start(signals["time_representation_path"].where(signals["sig_type"] != "spike_times"))
)

signals["end_time"] = signals["start_time"] + signals["duration"]
signals=signals.where(signals["duration"]>2)

@apply_file_func_decorator(".", name="resampled_normalize", out_folder=cache_path+"resampled_normalized", recompute=recompute)
def resample_normalize_data(arr, sig_type, new_fs):
    from xarray_helper import sampled_arr_from_events, resample_arr, normalize
    if sig_type=="spike_times":
        a, start = sampled_arr_from_events(arr.to_numpy() if isinstance(arr, xr.DataArray) else arr, new_fs)
        a=xr.DataArray(a, dims="t", coords=[start + (np.arange(a.size))/new_fs])
    else:
        a,counts = resample_arr(arr, "t", new_fs, return_counts=True, new_dim_name="t")
        a=a.where(counts==counts.max("t"), drop=True)
    a = normalize(a)
    a = a.assign_coords(fs=new_fs)
    return a

signals["resampled_continuous_path"] = resample_normalize_data(signals["time_representation_path"], signals["sig_type"], 250)


@apply_file_func_decorator(cache_path, name="cwt", out_folder=cache_path+"cwt", path_arg="debug_path", recompute=recompute)
def compute_cwt(arr, final_fs, debug_path):
    import pycwt
    from xarray_helper import resample_arr
    

    data, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(arr.to_numpy(), 1/arr["fs"].item(), s0=0.02, J=45)
    cwt = xr.DataArray(data=data, dims=["f", "t"], coords=dict(
        t=arr["t"],
        scales=("f", scales), f=("f", freqs),
        coi=("t", coi)
    ))

    final = resample_arr(cwt, "t", final_fs, new_dim_name="t")
    final["coi"] = resample_arr(cwt["coi"], "t", final_fs, new_dim_name="t")
    final = final.assign_coords(fs=final_fs)
    return final

signals["cwt_path"] = compute_cwt(signals["resampled_continuous_path"], 50)

@apply_file_func_decorator(cache_path, name="power_cwt", out_folder=cache_path+"power_cwt", recompute=recompute)
def compute_power_cwt(a: xr.DataArray):
    return np.abs(a * np.conj(a))

signals["cwt_power"] = compute_power_cwt(signals["cwt_path"])


@apply_file_func_decorator(cache_path, name="spectrogram", out_folder=cache_path+"spectrogram", path_arg="debug_path", recompute=recompute)
def compute_spectrogram(arr, final_fs, debug_path):
    import scipy.signal
    from xarray_helper import resample_arr
    
    window_size = int(arr["fs"].item()/final_fs)
    try:
        f,t,spectrogram = scipy.signal.spectrogram(arr.to_numpy(), arr["fs"].item(), nperseg=10*window_size, noverlap=9*window_size)
    except:
        print(window_size)
        raise
    spectrogram = xr.DataArray(data=spectrogram, dims=["f", "t"], coords=dict(
        t=t+arr["t"].min().item(),
        f=f,
    ))
    spectrogram = spectrogram.sel(f=slice(3, 50))
    final = resample_arr(spectrogram, "t", final_fs, new_dim_name="t")
    final = final.assign_coords(fs=final_fs)
    return final

signals["spectrogram"] = compute_spectrogram(signals["resampled_continuous_path"], 10)

@apply_file_func_decorator(cache_path, name="averaging along axis", n_ret=2, output_core_dims=[["f"], ["f"]], n_outcore_nans=46, save_group=cache_path+"averaging.pkl", recompute=recompute)
def average_along_axis(a: xr.DataArray, axis: str):
    r = a.where((a != np.inf) & (a != -np.inf)).mean(axis)
    res = r.to_numpy(), r["f"].to_numpy()
    return res

@apply_file_func_decorator(cache_path, name="averaging along axis", n_ret=2, output_core_dims=[["f"], ["f"]], n_outcore_nans=48, save_group=cache_path+"averaging_2.pkl", recompute=recompute)
def average_along_axis_2(a: xr.DataArray, axis: str):
    r = a.where((a != np.inf) & (a != -np.inf)).mean(axis)
    res = r.to_numpy(), r["f"].to_numpy()
    return res

pwelch, freqs = average_along_axis(signals["cwt_power"], "t")
pwelch["freqs"] = freqs
signals["pwelch_cwt"] = auto_remove_dim(pwelch.to_dataset(), kept_var=["freqs"])["cwt_power"]
signals["f"] = signals["freqs"]
signals = signals.drop_vars("freqs")

pwelch, freqs = average_along_axis_2(signals["spectrogram"], "t")
pwelch["freqs"] = freqs
signals["pwelch_spectrogram"] = auto_remove_dim(pwelch.rename(f="f2").to_dataset(), kept_var=["freqs"])["spectrogram"]
signals["f2"] = signals["freqs"]
signals = signals.drop_vars("freqs")


@apply_file_func_decorator(cache_path, name="tmp")
def show_arr(a):
    print(a)
    input()

pickle.dump(signals, open(cache_path+"signals_computed.pkl", "wb"))

logger.info("Creating signal pair dataset")
signals = signals.drop_dims(["f2"])
signal_pairs = xr.merge([signals.rename(**{x:f"{x}_1" for x in signals.coords if not x=="f"}, **{x:f"{x}_1" for x in signals.data_vars}), signals.rename(**{x:f"{x}_2" for x in signals.coords  if not x=="f"}, **{x:f"{x}_2" for x in signals.data_vars})])


def stack_dataset(dataset):
    dataset=dataset.copy()
    dataset["common_duration"] = xr.where(dataset["start_time_1"] > dataset["start_time_2"], 
             xr.where(dataset["end_time_1"] > dataset["end_time_2"],
                      dataset["end_time_2"]- dataset["start_time_1"], 
                      dataset["end_time_1"]- dataset["start_time_1"]
             ),
             xr.where(dataset["end_time_1"] > dataset["end_time_2"],
                      dataset["end_time_2"]- dataset["start_time_2"], 
                      dataset["end_time_1"]- dataset["start_time_2"]
             )
    )
    dataset["relevant_pair"] = (
        (dataset["Session_1"] == dataset["Session_2"])
        & (dataset["Contact_1"] != dataset["Contact_2"]) 
        & (dataset["sig_preprocessing_1"] <= dataset["sig_preprocessing_2"])
        & ((dataset["sig_preprocessing_1"] < dataset["sig_preprocessing_2"]) | (dataset["Contact_1"] < dataset["Contact_2"]))
        & (~dataset["resampled_continuous_path_1"].isnull())
        & (~dataset["resampled_continuous_path_2"].isnull())
        &  (dataset["common_duration"] >10)

        # & (dataset["Structure_1"] == dataset["Structure_2"])
        # & (dataset["sig_type_1"] =="bua")
        # & (dataset["sig_type_2"] =="spike_times")
    )
    dataset=dataset.stack(sig_preprocessing_pair=("sig_preprocessing_1","sig_preprocessing_2"), Contact_pair=("Contact_1", "Contact_2"))
    dataset = dataset.where(dataset["relevant_pair"].any("sig_preprocessing_pair"), drop=True)
    dataset = dataset.where(dataset["relevant_pair"].any("Contact_pair"), drop=True)
    return dataset

if not pathlib.Path(cache_path+"signal_pairs_xarray.pkl").exists():
    stack_size = 100
    signal_pairs_split = [signal_pairs.isel(dict(Contact_1=slice(stack_size*i, stack_size*(i+1)), Contact_2=slice(stack_size*j, stack_size*(j+1)))) 
                        for i in range(int(np.ceil(signal_pairs.sizes["Contact_1"]/stack_size)))
                        for j in range(int(np.ceil(signal_pairs.sizes["Contact_2"]/stack_size)))
    ]
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(stack_dataset, dataset) for dataset in signal_pairs_split]
        signal_pairs_split_stacked = [future.result() for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stacking")]
    signal_pairs_split_stacked: List[xr.Dataset] = [d.drop_vars(["Date_1", "Date_2"]) for d in signal_pairs_split_stacked]
    logger.info(f"Total number of Contact pairs is {int(np.sum([d.sizes['Contact_pair'] for d in signal_pairs_split_stacked]))} {int(np.max([d.sizes['sig_preprocessing_pair'] for d in signal_pairs_split_stacked]))}")
    signal_pairs = xr.merge(signal_pairs_split_stacked)
    logger.info("Dumping")
    pickle.dump(signal_pairs, open(cache_path+"signal_pairs_xarray.pkl", "wb"))
else:
    signal_pairs = pickle.load(open(cache_path+"signal_pairs_xarray.pkl", "rb"))
    
print(signal_pairs)

@apply_file_func_decorator(cache_path, name="coherence", out_folder=cache_path+"coherence", recompute=recompute, n=2)
def compute_coherence(resampled_1: xr.DataArray, resampled_2: xr.DataArray, final_fs):
    # print(resampled_1)
    # print(resampled_2)
    import pycwt
    from xarray_helper import resample_arr, normalize
    data= xr.merge([resampled_1.to_dataset(name="sig_1"), resampled_2.to_dataset(name="sig_2")], join="inner")
    # print("merge done")
    WCT, aWCT, coi, freqs, significance = pycwt.wct(normalize(data["sig_1"]).to_numpy(), normalize(data["sig_2"]).to_numpy(), 1/data["fs"].item(), s0=0.02, J=45, sig=False)
    # print("wct done")
    wct = xr.DataArray(data=WCT, dims=["f", "t"], coords=dict(
        t=data["t"],
        f=("f", freqs)
    ))
    phase = xr.DataArray(data=aWCT, dims=["f", "t"], coords=dict(
        t=data["t"],
        f=("f", freqs),
    ))
    coi = xr.DataArray(data=coi, dims=["t"], coords=dict(
        t=data["t"],
    ))
    r=xr.Dataset()
    r["wct"] = resample_arr(wct, "t", final_fs, new_dim_name="t")
    r["coi"] = resample_arr(coi, "t", final_fs, new_dim_name="t")
    r["phase"] = resample_arr(phase, "t", final_fs, new_dim_name="t")
    r = r.assign_coords(fs=final_fs)
    # print(r)
    # input()
    return r
    

# signal_pairs["coherence_path"] = compute_coherence(signal_pairs["resampled_continuous_path_1"].where(signal_pairs["relevant_pair"]), signal_pairs["resampled_continuous_path_2"].where(signal_pairs["relevant_pair"]), 50)

@apply_file_func_decorator(cache_path, name="averaging along axis_3", n_ret=2, output_core_dims=[["f"], ["f"]], n_outcore_nans=46, save_group=cache_path+"averaging_3.pkl", recompute=recompute)
def average_along_axis_3(a: xr.Dataset, axis: str):
    a =a["wct"]
    r = a.where((a != np.inf) & (a != -np.inf)).mean(axis)
    res = r.to_numpy(), r["f"].to_numpy()
    return res

# coherence, freqs = average_along_axis_3(signal_pairs["coherence_path"], "t")
# coherence["freqs"] = freqs
# signal_pairs["coherence_wct"] = auto_remove_dim(coherence.to_dataset(), kept_var=["freqs"])["coherence_path"]
# signal_pairs["f"] = signal_pairs["freqs"]
# signal_pairs = signal_pairs.drop_vars("freqs")


@apply_file_func_decorator(cache_path, name="coherence_scipy", n=2, n_ret=2, output_core_dims=[["f"], ["f"]], n_outcore_nans=48, save_group=cache_path+"coherence_scipy.pkl", recompute=recompute)
def compute_coherence_scipy(resampled_1: xr.DataArray, resampled_2: xr.DataArray, final_fs):
    import scipy.signal
    from xarray_helper import resample_arr, normalize
    try:
        data= xr.merge([resampled_1.to_dataset(name="sig_1"), resampled_2.to_dataset(name="sig_2")], join="inner")
    except:
        raise
    window_size = int(data["fs"].item()/final_fs)
    try:
        f,coherence_norm = scipy.signal.coherence(normalize(data["sig_1"]).to_numpy(), normalize(data["sig_2"]).to_numpy(), data["fs"].item(), nperseg=10*window_size, noverlap=9*window_size)
        f2, csd = scipy.signal.csd(normalize(data["sig_1"]).to_numpy(), normalize(data["sig_2"]).to_numpy(), data["fs"].item(), nperseg=10*window_size, noverlap=9*window_size)
    except:
        print(window_size)
        raise
    res = xr.Dataset()
    res["coherence_norm"] = xr.DataArray(data=coherence_norm, dims=["f"], coords=dict(f=f))
    res["coherence_phase"] = xr.DataArray(data=np.angle(csd), dims=["f"], coords=dict(f=f2))
    res["coherence"] = res["coherence_norm"] * np.exp(1j*res["coherence_phase"])
    res = res.sel(f=slice(3, 50))
    # f, axs = plt.subplots(1, 2)
    # axs[0].plot(data["t"], data["sig_1"], label="sig_1")
    # axs[0].plot(data["t"], data["sig_2"], label="sig_2")
    # axs[1].plot(res["f"], res, label="coherence")
    # plt.legend()
    # plt.show()
    return res["coherence"].to_numpy(), res["f"].to_numpy()


coherence, freqs = compute_coherence_scipy(signal_pairs["resampled_continuous_path_1"].where(signal_pairs["relevant_pair"]), signal_pairs["resampled_continuous_path_2"].where(signal_pairs["relevant_pair"]), 10)
coherence["freqs"] = freqs
signal_pairs["coherence_scipy"] = auto_remove_dim(coherence.rename(f="f2").to_dataset(name="coherence_scipy"), kept_var=["freqs"])["coherence_scipy"]
signal_pairs["f2"] = signal_pairs["freqs"]
signal_pairs = signal_pairs.drop_vars("freqs")

print(signal_pairs)
pickle.dump(signal_pairs, open(cache_path+"signal_pairs_computed.pkl", "wb"))

