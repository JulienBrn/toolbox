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
        cache_path = f"/media/julien/data1/JulienCache/{'All' if MODE=='ALL' else 'Balanced'}/"
        signals = xr.open_dataset(cache_path+"../signals.nc").load()
        metadata = xr.open_dataset(cache_path+"../metadata.nc")
        recompute=False
    case "SMALL":
        cache_path = "/media/julien/data1/JulienCache/Small/"
        signals = xr.open_dataset(cache_path+"../signals.nc").load().head(dict(Contact="50"))
        metadata = xr.open_dataset(cache_path+"../metadata.nc")
        recompute=False
recompute = recompute or force_recompute


signals["time_representation_path"] = xr.apply_ufunc(lambda x: np.where(x=="", np.nan, x), signals["time_representation_path"])
signals = signals.where(signals["Structure"].isin(["GPe", "STN", "STR"]), drop=True)
group_index_cols = ["Species", "Structure", "Healthy"]
signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
signals = signals.set_coords("group_index")
if MODE=="BALANCED":
    signals = signals.groupby("group_index").map(lambda x: x.head(20))
print(signals)

# grouped_results = xr.Dataset()
# transfert_coords = ["CorticalState", "FullStructure", "Condition"]

# for col in transfert_coords:
#     grouped_results[col] = signals[col].groupby("group_index").map(extract_unique, dim="Contact").unstack()
# grouped_results = auto_remove_dim(grouped_results, ignored_vars=["group_index"])
# grouped_results = grouped_results.set_coords(transfert_coords)




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
    f,t,spectrogram = scipy.signal.spectrogram(arr.to_numpy(), arr["fs"].item(), nperseg=10*window_size, noverlap=9*window_size)
    spectrogram = xr.DataArray(data=spectrogram, dims=["f", "t"], coords=dict(
        t=t+arr["t"].min().item(),
        f=f,
    ))
    spectrogram = spectrogram.sel(f=slice(3, 50))
    final = resample_arr(spectrogram, "t", final_fs, new_dim_name="t")
    final = final.assign_coords(fs=final_fs)
    return final

signals["spectrogram"] = compute_spectrogram(signals["resampled_continuous_path"], 50)

@apply_file_func_decorator(cache_path, name="averaging along axis", n_ret=2, output_core_dims=[["f"], ["f"]], n_outcore_nans=46, save_group=cache_path+"averaging.pkl", recompute=recompute)
def average_along_axis(a: xr.DataArray, axis: str):
    r = a.where((a != np.inf) & (a != -np.inf)).mean(axis)
    res = r.to_numpy(), r["f"].to_numpy()
    return res

@apply_file_func_decorator(cache_path, name="averaging along axis", n_ret=2, output_core_dims=[["f"], ["f"]], n_outcore_nans=10, save_group=cache_path+"averaging_2.pkl", recompute=recompute)
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
        & (dataset["Structure_1"] == dataset["Structure_2"])
        & (dataset["sig_type_1"] =="bua")
        & (dataset["sig_type_2"] =="spike_times")
        & (~dataset["resampled_continuous_path_1"].isnull())
        & (~dataset["resampled_continuous_path_2"].isnull())
        &  (dataset["common_duration"] >10)
    )
    dataset=dataset.stack(sig_preprocessing_pair=("sig_preprocessing_1","sig_preprocessing_2"), Contact_pair=("Contact_1", "Contact_2"))
    dataset = dataset.where(dataset["relevant_pair"].any("sig_preprocessing_pair"), drop=True)
    dataset = dataset.where(dataset["relevant_pair"].any("Contact_pair"), drop=True)
    return dataset

# print(signal_pairs)
# print((signal_pairs["relevant_pair"]).sum())
# print(signal_pairs["relevant_pair"].to_dataframe())
# exit()
# print((signal_pairs["relevant_pair"]).sum().item())
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
    signal_pairs_split_stacked = [d.drop_vars(["Date_1", "Date_2"]) for d in signal_pairs_split_stacked]
    signal_pairs = xr.merge(signal_pairs_split_stacked)
    pickle.dump(signal_pairs, open(cache_path+"signal_pairs_xarray.pkl", "wb"))
else:
    signal_pairs = pickle.load(open(cache_path+"signal_pairs_xarray.pkl", "rb"))

# signal_pairs_split_stacked = [stack_dataset(dataset) for dataset in tqdm.tqdm(signal_pairs_split, desc="Stacking")]

    
print(signal_pairs)
# exit()

# signal_pairs["sig_preprocessing_pair"] = signal_pairs["sig_preprocessing_1"] + signal_pairs["sig_preprocessing_2"]
# signal_pairs=signal_pairs.stack(sig_preprocessing_pair=("sig_preprocessing_1","sig_preprocessing_2"), Contact_pair=("Contact_1", "Contact_2"))
# signal_pairs["relevant_contact_pair"] = signal_pairs["relevant_pair"].any("sig_preprocessing_pair")
# signal_pairs["relevant_sigprocessing_pair"] = signal_pairs["relevant_pair"].any("Contact_pair")
# signal_pairs = signal_pairs.where(signal_pairs["relevant_contact_pair"], drop=True)
# signal_pairs = signal_pairs.where(signal_pairs["relevant_sigprocessing_pair"], drop=True)
# logger.info("Done creating signal pair dataset")
# print(signal_pairs)

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
    

# input("Waiting to compute coherence")
signal_pairs["coherence"] = compute_coherence(signal_pairs["resampled_continuous_path_1"].where(signal_pairs["relevant_pair"]), signal_pairs["resampled_continuous_path_2"].where(signal_pairs["relevant_pair"]), 50)
print(signal_pairs)
pickle.dump(signal_pairs, open(cache_path+"signal_pairs_computed.pkl", "wb"))
exit()

# show_arr(signals["spectrogram"])
# show_arr(signals["cwt_power"])


plt.plot(signals["f2"], signals["pwelch_spectrogram"].mean(["Contact", "sig_preprocessing"])*100)
plt.plot(signals["f"], signals["pwelch_cwt"].mean(["Contact", "sig_preprocessing"]))
plt.show()
exit()
# signals=signals.head(100)
signals["cwt_path"] = compute_cwt(signals["resampled_continuous_path"], 50)

def compute_spectrogram(arr, sig_type, final_fs, debug_path):
    pass

def compute_cwt_power(arr: xr.Dataset):
    pass
signals["cwt_power"] = lambda a: np.abs(a * np.conj(a))
print(signals)
exit()


def compute_cwt(a, sig_type, pre_cwt_fs, final_fs, plot=False, paths=[]):
    import pycwt
    from toolbox import sampled_arr_from_events, resample_arr
    if plot:
        f,[ax_in, ax_out] = plt.subplots(2,sharex=True)
        f.suptitle(f"Computation of wavelets for {paths}", size="small")
        ax_in_leg=[]
    if sig_type=="spike_times":
        if plot:
            ax_ev = ax_in.twinx()
            ax_in_leg=ax_in_leg+ax_ev.eventplot(a, color="pink",label="spike times")
        a, start = sampled_arr_from_events(a, pre_cwt_fs)
        a=xr.DataArray(a, dims="t", coords=[start + (np.arange(a.size))/pre_cwt_fs])
    else:
        a,counts = resample_arr(a, "t", pre_cwt_fs, return_counts=True, new_dim_name="t")
        a=a.where(counts==counts.max("t"), drop=True)
    if plot:
        ax_in.set_title(f"Input data before normalization downsampled to {pre_cwt_fs}")
        ax_in_leg=ax_in_leg+a.plot(ax=ax_in, label="Resampled data given to cwt before z-score")
        # print(type(ax_in_leg[0]))
        ax_in.legend(ax_in_leg, [l.get_label() for l in ax_in_leg])
    data, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(normalize(a).to_numpy(), 1/pre_cwt_fs, s0=0.02, J=45)
    res = xr.DataArray(data=data, dims=["f", "t"], coords=dict(
        t=a["t"],
        scales=("f", scales), f=("f", freqs),
        coi=("t", coi)
    ))
    # print(res)
    unresampled=res
    res = resample_arr(res, "t", final_fs, new_dim_name="t")
    res["coi"] = resample_arr(unresampled["coi"], "t", final_fs, new_dim_name="t")
    # print(res)
    
    if plot:
        ax_out.set_title(f"Outputtime/freq representation downsampled to {final_fs}")
        np.abs(res).plot(ax=ax_out, add_colorbar=False)
        (1/res["coi"]).plot(ax=ax_out, color="red", label="cone of influence")
        # axs[1].plot(res["t"], 1/res["coi"], color="red")
        ax_out.legend()
        f.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    return res











print(np.abs(signals["duration"].sel(sig_preprocessing="bua")-signals["duration"].sel(sig_preprocessing="lfp")).max().item())
print((signals["duration"].sel(sig_preprocessing="bua")-signals["duration"].sel(sig_preprocessing="neuron_0")).max().item())
print((signals["duration"].sel(sig_preprocessing="bua")-signals["duration"].sel(sig_preprocessing="neuron_0")).min().item())

exit()
signals["bua_duration"] = apply_file_func(lambda arr: arr["index"].max() - arr["index"].min(), ".", signals["time_representation_path"].sel(sig_preprocessing="bua"), name="duration", save_group=cache_path+"durations.pkl")
signals["spike_duration"] = apply_file_func(lambda arr: float(np.max(arr) - np.min(arr)), ".", signals["time_representation_path"].where(signals["sig_type"] == "spike_times"), name="spike_duration", save_group=cache_path+"spike_durations.pkl")
signals["n_spikes"] = apply_file_func(lambda arr: arr.size, ".", signals["time_representation_path"].where(signals["sig_type"] == "spike_times"), name="n_spike", save_group=cache_path+"n_spikes.pkl")
signals["n_spikes/s"] = signals["n_spikes"]/signals["spike_duration"]
signals["n_data_points"] = apply_file_func(lambda arr: float(arr.size), ".", signals["time_representation_path"], name="n_datapoints", save_group=cache_path+"n_datapoints.pkl")
signals["_diff"] = (signals["bua_duration"] - signals["spike_duration"])

exit()




    
def compute_cwt(a, sig_type, pre_cwt_fs, final_fs, plot=False, paths=[]):
    import pycwt
    if plot:
        f,[ax_in, ax_out] = plt.subplots(2,sharex=True)
        f.suptitle(f"Computation of wavelets for {paths}", size="small")
        ax_in_leg=[]
    if sig_type=="spike_times":
        if plot:
            ax_ev = ax_in.twinx()
            ax_in_leg=ax_in_leg+ax_ev.eventplot(a, color="pink",label="spike times")
        a, start = sampled_arr_from_events(a, pre_cwt_fs)
        a=xr.DataArray(a, dims="t", coords=[start + (np.arange(a.size))/pre_cwt_fs])
    else:
        a,counts = resample_arr(a, "t", pre_cwt_fs, return_counts=True, new_dim_name="t")
        a=a.where(counts==counts.max("t"), drop=True)
    if plot:
        ax_in.set_title(f"Input data before normalization downsampled to {pre_cwt_fs}")
        ax_in_leg=ax_in_leg+a.plot(ax=ax_in, label="Resampled data given to cwt before z-score")
        # print(type(ax_in_leg[0]))
        ax_in.legend(ax_in_leg, [l.get_label() for l in ax_in_leg])
    data, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(normalize(a).to_numpy(), 1/pre_cwt_fs, s0=0.02, J=45)
    res = xr.DataArray(data=data, dims=["f", "t"], coords=dict(
        t=a["t"],
        scales=("f", scales), f=("f", freqs),
        coi=("t", coi)
    ))
    # print(res)
    unresampled=res
    res = resample_arr(res, "t", final_fs, new_dim_name="t")
    res["coi"] = resample_arr(unresampled["coi"], "t", final_fs, new_dim_name="t")
    # print(res)
    
    if plot:
        ax_out.set_title(f"Outputtime/freq representation downsampled to {final_fs}")
        np.abs(res).plot(ax=ax_out, add_colorbar=False)
        (1/res["coi"]).plot(ax=ax_out, color="red", label="cone of influence")
        # axs[1].plot(res["t"], 1/res["coi"], color="red")
        ax_out.legend()
        f.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    return res

def compute_coherence(a: xr.DataArray,b: xr.DataArray,ta: xr.DataArray, tb: xr.DataArray, s1, s2, pre_cwt_fs, final_fs, plot=True):
    import pycwt
    if plot:
        f,[ax_a, ax_b, ax_out, ax_out_phase] = plt.subplots(4,sharex=True)
        ax_a.set_title(f"cwt for a")
        ax_b.set_title(f"cwt for b")
        np.abs(a).plot(ax=ax_a, add_colorbar=False)
        np.abs(b).plot(ax=ax_b, add_colorbar=False)
        (1/a["coi"]).plot(ax=ax_a, color="red", label="cone of influence")
        (1/b["coi"]).plot(ax=ax_b, color="red", label="cone of influence")
    
    # a.name = "wa"
    # b.name="wb"
    # a = a.rename(dict(coi="coi_a"))
    # b = b.rename(dict(coi="coi_b"))

    # r = xr.merge([a, b], join="inner")

    ############### This method does not work..... #####################

    # r["coi"] = np.minimum(r["coi_a"], r["coi_b"])
    # r = r.drop_vars(["coi_a", "coi_b"])
    # r["wab"] = (r["wa"] * np.conj(r["wb"]))

    # wavelet = pycwt.wavelet._check_parameter_wavelet("morlet")
    # S1 = wavelet.smooth(np.abs((r["wa"]) ** 2 / r["scales"]).to_numpy(), 0.02, 0.02, r["scales"].to_numpy())
    # S2 = wavelet.smooth(np.abs((r["wb"]) ** 2 / r["scales"]).to_numpy(), 0.02, 0.02, r["scales"].to_numpy())
    # S12 = wavelet.smooth((np.abs(r["wab"]) / r["scales"]).to_numpy(), 0.02, 0.02, r["scales"].to_numpy())
    # r["wct"] = xr.DataArray(data=np.abs(S12) ** 2 / (S1 * S2), dims=("f", "t"))
    
    ############### Using basic method ############################
    # print(ta)
    # print(tb)
    amin, amax = (ta["t"].min(), ta["t"].max()) if not s1=="spike_times" else (np.min(ta), np.max(ta))
    bmin, bmax = (tb["t"].min(), tb["t"].max()) if not s2=="spike_times" else (np.min(tb), np.max(tb))
    if amin > bmax or bmin > amax:
        return np.nan
    ars = []
    for a, sig_type in zip([ta, tb], [s1, s2]):
        if sig_type=="spike_times":
            a, start = sampled_arr_from_events(a, pre_cwt_fs)
            a=xr.DataArray(a, dims="t", coords=[start + (np.arange(a.size))/pre_cwt_fs])
        else:
            a,counts = resample_arr(a, "t", pre_cwt_fs, return_counts=True, new_dim_name="t")
            a=a.where(counts==counts.max("t"), drop=True)
        a.name = ["a", "b"][len(ars)]
        ars.append(a)
    r2 = xr.merge(ars, join="inner")
    if r2.sizes["t"] <10:
        # print(ars[0])
        # print(ars[1])
        # print(r2)
        # print("returning nan")
        # exit()
        return np.nan
    WCT, aWCT, coi, freqs, significance = pycwt.wct(r2["a"].to_numpy(), r2["b"].to_numpy(), 1/pre_cwt_fs, s0=0.02, J=45, sig=False)
    wct = xr.DataArray(data=WCT, dims=["f", "t"], coords=dict(
        t=r2["t"],
        f=("f", freqs)
    ))
    phase = xr.DataArray(data=aWCT, dims=["f", "t"], coords=dict(
        t=r2["t"],
        f=("f", freqs),
    ))
    coi = xr.DataArray(data=coi, dims=["t"], coords=dict(
        t=r2["t"],
    ))
    r=xr.Dataset()
    r["wct"] = resample_arr(wct, "t", final_fs, new_dim_name="t")
    r["coi"] = resample_arr(coi, "t", final_fs, new_dim_name="t")
    r["phase"] = resample_arr(phase, "t", final_fs, new_dim_name="t")
    if plot:
        r["wct"].plot(ax=ax_out, add_colorbar=False)
        (1/r["coi"]).plot(ax=ax_out, color="red", label="cone of influence")
        r["phase"].plot(ax=ax_out_phase, add_colorbar=False)
        (1/r["coi"]).plot(ax=ax_out_phase, color="red", label="cone of influence")
        f.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    
    # r = r.drop_vars(["coi_a", "coi_b", "wa", "wb"])
    return r

    # a = a.to_dataset(name="a")
    # b = b.to_dataset(name="b")
    # print(a)
    # print(b)
    # a["t"] = xr.apply_ufunc(lambda x: np.round(x.left*50+0.00001)/50, a["t"], vectorize=True)
    # b["t"] = xr.apply_ufunc(lambda x: np.round(x.left*50+0.00001)/50, b["t"], vectorize=True)
    # tmp=xr.merge([a, b], join="inner")
    # a = tmp["a"]
    # b=tmp["b"]
    # res = np.abs(a*np.conj(b))**2 / ((np.abs(a)**2)*(np.abs(b)**2))
    # print(res.to_dataset(name="coherence"))
    # input()
    # return res

def test(a):
    from regular_index import RegularIndex
    # print(a)
    a = a.drop_indexes(["t"])
    a = a.set_xindex(["t"], RegularIndex)
    # a = a.reset_coords("t")
    print(a)
    # print(len(a.indexes))
    exit()

def normalize(a: xr.DataArray):
    std = a.std()
    a_normal = (a - a.mean()) / std
    return a_normal

def rotate_test(a: xr.DataArray, arr:str=None):
    if not arr is None:
        a = a[arr]
    # print(a)
    r = a.where((a != np.inf) & (a != -np.inf)).mean("t")
    res = r.to_numpy(), r["f"].to_numpy()
    # print(res)
    # input()
    return res
# print(signals.indexes)
# signals["test"] = apply_file_func(test, ".", signals["time_representation_path"])
signals["cwt_path"] = apply_file_func(compute_cwt, ".", signals["time_representation_path"], signals["sig_type"], 250, 50, out_folder="./cwt", name="cwt", path_arg="paths")
signals["time_freq_repr"] = apply_file_func(lambda a: np.abs(a * np.conj(a)), ".", signals["cwt_path"], out_folder="./tf_repr2", name="time_freq_representation")
pwelch, freqs = apply_file_func(rotate_test, ".", signals["time_freq_repr"], name="rotate_test", n_ret=2, output_core_dims=[["f"], ["f"]], save_group="pwelch_sig.pkl")
pwelch["freqs"] = freqs
signals["pwelch"] = auto_remove_dim(pwelch.to_dataset(), kept_var=["freqs"])["time_freq_repr"]
signals["f"] = signals["freqs"]
signals = signals.drop_vars("freqs")
def rm_tuple_nan(x):

    res = np.nan if isinstance(x, tuple) and len(x) ==1 and pd.isna(x[0]) else x
    # print(x, res, len(x))
    return res
signals = xr.apply_ufunc(rm_tuple_nan, signals, vectorize=True)
# print(test)
# exit()
# print(signals)
# exit()

print(signals)
if not pathlib.Path("pair_signals.pkl").exists():
# if True:
    defined_signals = signals[["cwt_path", "time_representation_path"]].to_dataframe().reset_index()
    defined_signals = defined_signals.loc[~pd.isna(defined_signals["cwt_path"])]
    defined_signals = defined_signals.drop(columns="has_entry")
    defined_signals = defined_signals.drop(columns=["group_index"])
    defined_signals = defined_signals.sort_values("sig_preprocessing")
    pair_signals = toolbox.group_and_combine(defined_signals, ["Session", "FullStructure"], include_eq=False)
    pair_signals = (pair_signals.loc[pair_signals["Contact_1"] != pair_signals["Contact_2"]]).copy()

    # pair_signals["Contact_1_old"] = pair_signals["Contact_1"]
    # pair_signals["Contact_2_old"] = pair_signals["Contact_2"]
    # pair_signals["Contact_1"] = np.where(pair_signals["sig_preprocessing_1"] < pair_signals["sig_preprocessing_2"], pair_signals["Contact_1_old"], pair_signals["Contact_2_old"])
    # pair_signals["Contact_2"] = np.where(pair_signals["sig_preprocessing_1"] < pair_signals["sig_preprocessing_2"], pair_signals["Contact_2_old"], pair_signals["Contact_1_old"])
    
    # print(pair_signals.columns)
    # exit()
    
    # pair_signals = pair_signals.drop(columns=["Contact_1_old", "Contact_2_old"])
    pair_signals["Contact_pair"] = pd.MultiIndex.from_arrays([pair_signals["Contact_1"], pair_signals["Contact_2"]], names=["Contact_1", "Contact_2"])
    pair_signals = pair_signals.set_index(["Contact_pair", "sig_preprocessing_1", "sig_preprocessing_2"])
    pair_signals.to_csv("pair_signals.tsv", sep="\t")
    
    pair_signals = xr.Dataset.from_dataframe(pair_signals)
    for coord in signals.coords:
        if coord in pair_signals:
            continue
        if coord+"_1" in pair_signals and coord+"_2" in pair_signals:
            coord = coord+"_1"
            coord2 = coord.replace("_1", "_2")
            if ((pair_signals[coord] == pair_signals[coord2]) | (pd.isna(pair_signals[coord])) | (pd.isna(pair_signals[coord2]))).all():
                pair_signals = pair_signals.drop_vars(coord2)
                pair_signals = pair_signals.rename({coord:coord.replace("_1", "")})
        elif coord+"_1" in pair_signals:
            raise Exception("Strange")
    
    print(pair_signals)
    pair_signals = auto_remove_dim(pair_signals, ignored_vars=["Contact_pair"])
    pickle.dump(pair_signals, open("pair_signals.pkl", "wb"))
else:
    pair_signals = pickle.load(open("pair_signals.pkl", "rb"))
# print(pair_signals)
pair_signals["has_entry_1"] = xr.apply_ufunc(lambda x: ~pd.isna(x), pair_signals["cwt_path_1"])
pair_signals["has_entry_2"] = xr.apply_ufunc(lambda x: ~pd.isna(x), pair_signals["cwt_path_2"])
pair_signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([pair_signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact_pair'], coords=[pair_signals["Contact_pair"]])
 
pair_signals = pair_signals.set_coords([v for v in pair_signals.variables if not "cwt_path" in v and not "time_representation_path" in v])
pair_signals =pair_signals.where(((pair_signals["sig_type_1"] == "bua") & (pair_signals["sig_type_2"] == "spike_times")) | ((pair_signals["sig_type_2"] == "bua") & (pair_signals["sig_type_1"] == "spike_times")))
# print(pair_signals)



# a= 0.036004337535534746+0.018279389420020897j
# b = 98.23709725843953-769.0045991853731j
# print(np.abs(a)**2)
# print(np.abs(b)**2)
# print(np.abs(a*np.conj(b))**2)
# print((np.abs(a*np.conj(b))**2) / ((np.abs(a)**2)*(np.abs(b)**2)))
# exit()

def compute_inter_length(ta, tb, s1, s2):
    amin, amax = (ta["t"].min(), ta["t"].max()) if not s1=="spike_times" else (np.min(ta), np.max(ta))
    bmin, bmax = (tb["t"].min(), tb["t"].max()) if not s2=="spike_times" else (np.min(tb), np.max(tb))
    if amin > bmax or bmin > amax:
        return 0
    else:
        return min(amax,bmax) - max(amin, bmin)

    
pair_signals["intersection_length"] = apply_file_func(compute_inter_length, ".",
    pair_signals["time_representation_path_1"], pair_signals["time_representation_path_2"],
    pair_signals["sig_type_1"], pair_signals["sig_type_2"],
    n=2, save_group="./intersection_length.pkl", name="intersection_length")

# debug = pair_signals["intersection_length"].where(pair_signals["Species"]=="Rat", drop=True).where(pair_signals["Structure"]=="STR", drop=True).where(pair_signals["Healthy"]=="True", drop=True)
# print(debug)
# exit()
pair_signals["coherence"] = apply_file_func(compute_coherence, ".", 
    pair_signals["cwt_path_1"].where(pair_signals["intersection_length"] > 5), pair_signals["cwt_path_2"].where(pair_signals["intersection_length"] > 5), 
    pair_signals["time_representation_path_1"].where(pair_signals["intersection_length"] > 5), pair_signals["time_representation_path_2"].where(pair_signals["intersection_length"] > 5),
    pair_signals["sig_type_1"], pair_signals["sig_type_2"],
    250, 50, False,
    out_folder="./coherence", name="coherence", n=4,
    save_group="./all_coherence.pkl")

# print(pair_signals["coherence"])
# exit()
coherence_mean, freqs = apply_file_func(lambda x: rotate_test(x, "wct"), ".", pair_signals["coherence"], name="rotate_test", n_ret=2, output_core_dims=[["f"], ["f"]], save_group="coherence_mean.pkl")
coherence_mean["freqs"] = freqs
pair_signals["coherence_mean"] = auto_remove_dim(coherence_mean.to_dataset(), kept_var=["freqs"])["coherence"]

pair_signals["f"] = pair_signals["freqs"]
pair_signals = pair_signals.drop_vars("freqs")


print(pair_signals)
print(pair_signals["coherence_mean"].count())
# print(pair_signals.groupby(["Session"]).apply(lambda d: d.groupby(["Contact_1"]).ngroup()).max())
# print(pair_signals.groupby(["Session"]).apply(lambda d: d.groupby(["Contact_2"]).ngroup()).max())
# print(pair_signals.groupby(["Session"]).ngroup().max())
# print(pair_signals.groupby(["Session", "Contact_2", "sig_preprocessing_2"]).ngroup())
# signals.groupby("Session").map(lambda a: a.merge(a.rename({d:f"{d}_other" for d in list(a.dims) + list(a.coords) + list(a.variables)})))
# print(pair_signals)
# print(pair_signals.coords)
# print(pair_signals.groupby("group_index").count().unstack())
# exit()


####### Finally, let's do some stats ##############
grouped_results["n_sessions"] = signals["Session"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda x: np.unique(x).size, a, input_core_dims=[a.dims])).unstack()
grouped_results["ContactCounts"] = signals["has_entry"].any(dim="sig_preprocessing").groupby("group_index").count().unstack()
grouped_results["avg_NeuronCountsPerContact"] = signals["has_entry"].where(~(signals["sig_preprocessing"].isin(["lfp", "bua"])), drop=True).sum(dim="sig_preprocessing").groupby("group_index").mean().unstack()
grouped_results["max_neuron_in_session"] = signals["has_entry"].where(~(signals["sig_preprocessing"].isin(["lfp", "bua"])), drop=True).sum(dim="sig_preprocessing").groupby("group_index").map(lambda a: a.groupby("Session").sum().max()).unstack()
grouped_results["avg_bua_duration"] = signals["bua_duration"].groupby("group_index").map(lambda a: a.mean()).unstack()
grouped_results["avg_spike_duration"] = signals["spike_duration"].groupby("group_index").map(lambda a: a.mean()).unstack()
grouped_results["avg_n_spike/s"] = signals["n_spikes/s"].groupby("group_index").map(lambda a: a.mean()).unstack()
grouped_results["pwelch"] = signals["pwelch"].groupby("group_index").mean().unstack().groupby("sig_type").mean()
grouped_results["coherence_mean"] = pair_signals["coherence_mean"].groupby("group_index").mean().unstack().groupby("sig_type_1").mean().groupby("sig_type_2").mean()
grouped_results["pwelch2"] = signals.groupby("group_index").map(lambda a: a["pwelch"].weighted(a["bua_duration"]).mean("Contact")).unstack().groupby("sig_type").mean()
grouped_results["CoherencePairCounts"] = (pair_signals["intersection_length"] > 5).groupby("group_index").map(lambda a: a.sum()).unstack()
print(grouped_results)

# print(grouped_results["CoherencePairCounts"])
# exit()
# print(grouped_results["pwelch"])
# print(grouped_results["pwelch2"])
# exit()
def get_values(a):
    global all_progress
    all_progress.update(1)
    grp_index = str(a["group_index"].to_numpy()[0])
    folder = pathlib.Path(f"./pwelch_gathered/{grp_index}")
    if (folder/"table.pkl").exists():
        return pickle.load((folder/"table.pkl").open("rb"))
    vals = {}
    progress = tqdm.tqdm(desc=f"  Gathering {grp_index}", total=float(a.count()))
    expected_coords = np.arange(5, 50)
    def compute(ar, contact):
        if pd.isna(ar):
            return xr.DataArray(data=np.ones_like(expected_coords)*np.nan, dims="f", coords=[expected_coords])
        ar: xr.DataArray = pickle.load(open(ar, "rb"))
        ar = ar.expand_dims({"Contact": [contact]})
        # print(ar)
        # exit()
        progress.update(1)
        coords = ar["f"].to_numpy()
        if (expected_coords != coords).any():
            raise Exception("Unmatching coordinates")
        arrays = np.ndarray(coords.size, dtype=object)
        for i in range(coords.size):
            arrays[i] = ar.isel(f=i, drop=True)
            arrays[i]["t"] = xr.apply_ufunc(lambda x: x.left, arrays[i]["t"], vectorize=True)
        ar = xr.DataArray(data=arrays, dims="f", coords=[coords])
        return ar
    arrays: xr.DataArray = xr.apply_ufunc(compute, a, a["Contact"], output_core_dims=[["f"]], vectorize=True)
    arrays["f"] = expected_coords
    # print(arrays)
    progress = tqdm.tqdm(desc=f"  Merging {grp_index}", total=float(arrays.size/arrays.sizes["Contact"]))
    def merge(ars, path):
        progress.update(1)
        progress.set_postfix_str(f"{path}, stacking {ars.size}")
        file: pathlib.Path = folder/(path+".pkl")
        if file.exists():
            return str(file)
        try:
            ars = [ar.stack(Window=["t", "Contact"]) for ar in ars if not isinstance(ar, float)]
        except Exception as e:
            e.add_note(f"ars[0] = {ars[0], type(ars[0])}\nars was\n{ars}")
            raise e
        if len(ars) == 0:
            return np.nan
        progress.set_postfix_str(f"{path}, concatenating {len(ars)}")
        coords = {k: ("Window", np.concatenate([ar[k] for ar in ars])) for k in ars[0].coords if not k=="Window"}
        data=np.concatenate(ars)
        progress.set_postfix_str(f"{path}, creating {data.shape}")
        ars = xr.DataArray(data=data, dims="Window", coords=coords)
        
        file.parent.mkdir(exist_ok=True, parents=True)
        # print(f"Writing {file}")
        progress.set_postfix_str(f"{path}, dumping {ars.shape}")
        print(ars)
        exit()
        pickle.dump(ars, file.open("wb"))
        # xr.concat(ars, dim="Window")
        # print("Finally")
        return str(file)
    arrays["file_path"] = "freq_" + arrays["f"].astype(str).astype(object) +"/"+ arrays["sig_preprocessing"]
    # print(arrays)
    res = xr.apply_ufunc(merge, arrays, arrays["file_path"], input_core_dims=[["Contact"], []], vectorize=True)
    folder.mkdir(exist_ok=True, parents=True)
    pickle.dump(res, (folder/"table.pkl").open("wb"))
    return res
    # print(arrays)
    # exit()
    
    res = {}
    for f in tqdm.tqdm(vals.keys(), desc="Dumping"):
        ar = np.concatenate(vals[f])
        ar = xr.DataArray(data=ar, dims="window")
        (folder/f"freq_{f}/data.pkl").parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(ar,  (folder/f"freq_{f}/data.pkl").open("wb"))
        res[f] = str((folder/f"freq_{f}/data.pkl"))
    # print({f: [ar.size for ar in ars] for f, ars in vals.items()})
    # vals = {f:np.concatenate(ars) for f,ars in vals.items()}
    # vals = {f: xr.DataArray(data=v, dims="window") for f,v in vals.items()}
    # for f,v in tqdm.tqdm(vals.items(), desc="Dumping"):
    #     (folder/f"freq_{f}/data.pkl").parent.mkdir(exist_ok=True, parents=True)
    #     pickle.dump(v,  (folder/f"freq_{f}/data.pkl").open("wb"))
    # vals = {f: str((folder/f"freq_{f}/data.pkl")) for f,v in vals.items()}
    res = pd.Series(res)
    # print(vals)
    # print({f: ar.size for f, ar in vals.items()})
    # print(np.array(list(vals.values())).shape)
   
    res = xr.DataArray.from_series(res).rename(index="f")
    pickle.dump(res, (folder/"table.pkl").open("wb"))
    # print(res)
    # exit()
    return res

# all_progress = tqdm.tqdm(desc="Extracting", total=float(len(signals["time_freq_repr"].groupby("group_index"))))
# grouped_results["time_freq_repr_values"] = signals["time_freq_repr"].groupby("group_index").map(get_values).unstack()

# grouped_results["pwelch"] = apply_file_func(lambda x: x.mean(), ".", grouped_results["time_freq_repr_values"], save_group="./pwelch.pkl", name="welch")

print(grouped_results)

# welch = grouped_results["pwelch"].plot(x="f", hue="Species", style="Healthy", row="Structure", col="sig_preprocessing")
# plt.show()



# print(signals)
# signals.to_dataframe().to_csv("tmp.csv")
# # grouped_results["DurationDiff"] = 
grouped_results["DurationDiffBorders"] = signals["_diff"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[-np.inf, -0.001, 0.1, 1, 10, 100, np.inf])[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
grouped_results["NBDurationDiff"] = signals["_diff"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[-np.inf, -0.001, 0.1, 1, 10, 100, np.inf])[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()

# grouped_results["n_spikes_borders"] = signals["n_spikes"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 5, 20, 50, 100, 500, np.inf])[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
# grouped_results["NB_n_spikes"] = signals["n_spikes"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 5, 20, 50, 100, 500, np.inf])[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
# grouped_results["n_spikes/s_borders"] = signals["n_spikes/s"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 1, 5, 10, 20, 50, np.inf])[1][1:], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
# grouped_results["NB_n_spikes/s"] = signals["n_spikes/s"].groupby("group_index").map(lambda a: xr.apply_ufunc(lambda a: np.histogram(a, bins=[0, 1, 5, 10, 20, 50, np.inf])[0], a, input_core_dims=[a.dims], output_core_dims=[["bins"]])).unstack()
# print(grouped_results)
# print(np.abs(metadata["Duration"] - signals["Duration"]).max())
# # print(signals)
# errors = signals.where(np.abs(signals["_diff"]) >10, drop=True).merge(metadata["signal_file_source"], join="left").to_dataframe()
# print(errors)
# errors.to_csv("errors.csv")
# print(errors.loc[errors["signal_file_source"]=="File(CTL/A1/20051207/a07/GPe/Raw.mat)[pjx301a_Probe2, values]", :])
# print(metadata)
# print(grouped_results)
pwelch_df = grouped_results["pwelch"].to_dataframe()
coherence_mean_df = grouped_results["coherence_mean"].sel(Structure="GPe").to_dataframe()
pwelch2_df = grouped_results["pwelch2"].to_dataframe()
grouped_results = grouped_results.drop_dims(["bins", "f", "sig_type", "sig_type_1", "sig_type_2"])
# print(grouped_results.drop_vars(["CorticalState", "FullStructure", "Condition"]).to_dataframe().to_string())

# print(grouped_results.to_dataframe())

##### Now, let's plot the data

basic_data = grouped_results.to_array("info", name="counts").to_dataframe().sort_index(level=["Species", "Structure", "Healthy"]).reset_index("Healthy", drop=True).set_index("Condition", append=True)
maxes=basic_data["counts"].groupby("info").max()
# basic_data["counts"] = basic_data["counts"]/maxes
basic_data["group"] = [str(x[1:]) for x in basic_data.index.values]

import seaborn as sns

if False:
# xr.plot.pcolormesh(signals["pwelch"].sel(sig_preprocessing="bua"),y="Contact", x="f")
    psig_data = signals["pwelch"].to_dataframe().reset_index()
    psig_data = psig_data[(psig_data["f"] > 6) & (psig_data["f"] < 45)].copy()
    psig_data["SSH"] = psig_data["Species"].astype(str) + psig_data["Structure"].astype(str) + psig_data["Healthy"].astype(str)
    psig_data = psig_data.groupby(["Contact", "f", "sig_type", "SSH", "Species", "Structure", "Healthy"])["pwelch"].mean().reset_index()
    psig_data["sig_type"] = np.where(psig_data["sig_type"].str.contains("spike"), "spike", psig_data["sig_type"])
    # psig_data = psig_data.sort_values("pwelch")
    # print(psig_data["SSH"])
    # psig_data = psig_data[psig_data["SSH"].isin(["RatGPe0"])]
    print(psig_data.shape)
    print(psig_data.columns)
    pwelch_sig_fig = toolbox.FigurePlot(psig_data, figures="Species", col="SSH", row="sig_type", sharey=False, margin_titles=True)
    pwelch_sig_fig.pcolormesh(x="f", y="Contact", value="pwelch", ysort=20.0, ylabels=False)



coherence_data = pair_signals["coherence_mean"].to_dataframe().reset_index()
coherence_data = coherence_data[(coherence_data["f"] > 6) & (coherence_data["f"] < 45) & (coherence_data["coherence_mean"] < 10**9)].copy()
# print(coherence_data)
# print(coherence_data[~coherence_data["coherence_mean"].isna()])
# print(coherence_data[coherence_data["coherence_mean"] > 10**9])
coherence_data["SSH"] = coherence_data["Species"].astype(str) + coherence_data["Structure"].astype(str) + coherence_data["Healthy"].astype(str)
coherence_data = coherence_data.groupby(["Contact_pair", "f", "sig_type_1", "sig_type_2", "SSH", "Species", "Structure", "Healthy"])["coherence_mean"].mean().reset_index()
print(coherence_data[~coherence_data["coherence_mean"].isna()])

# print("SHOWING")
# print(coherence_data[["sig_type_1", "sig_type_2"]].drop_duplicates())
# coherence_data = coherence_data.loc[(coherence_data["sig_type_1"] == coherence_data["sig_type_2"])].copy()
coherence_data["sig_type"] = coherence_data["sig_type_1"] + coherence_data["sig_type_2"]
print(coherence_data)
coherence_sig_fig = toolbox.FigurePlot(coherence_data, figures="Species", col="SSH", row="sig_type", sharey=False, margin_titles=True)
coherence_sig_fig.pcolormesh(x="f", y="Contact_pair", value="coherence_mean", ysort=20.0, ylabels=False)


# g = sns.FacetGrid(data=basic_data.reset_index(), col="info", col_wrap=3, sharex=False, hue="Species", aspect=2).map_dataframe(sns.barplot, x="counts", y="group").tight_layout().add_legend()
# for ax in g.axes.flat:
#     ax.set_ylabel(None)
# g.figure.subplots_adjust(top=.9, bottom=0.05)

# pwelch_fig = sns.relplot(kind="line", data=pwelch_df, x="f", y="pwelch", col="Structure", row="sig_type", hue="Species", style="Healthy", facet_kws=dict(margin_titles=True, sharey="row"))
# pwelch_fig.figure.subplots_adjust(top=.9, bottom=0.05)
# pwelch2_fig = sns.relplot(kind="line", data=pwelch2_df, x="f", y="pwelch2", col="Structure", row="sig_type", hue="Species", style="Healthy", facet_kws=dict(margin_titles=True, sharey="row"))
# pwelch2_fig.figure.subplots_adjust(top=.9, bottom=0.05)
# coherence_fig = sns.relplot(kind="line", data=coherence_mean_df, x="f", y="coherence_mean", col="sig_type_2", row="sig_type_1", hue="Species", style="Healthy", facet_kws=dict(margin_titles=True, sharey="row"))
# coherence_fig.figure.subplots_adjust(top=.9, bottom=0.05)


# sns.barplot(data=basic_data, x = [str(x) for x in basic_data.index.values],  y="counts", hue="info", gap=0)
# plt.title(str(maxes))
plt.show()