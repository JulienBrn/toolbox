import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave
from xarray_helper import apply_file_func, auto_remove_dim, nunique


logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
tqdm.tqdm.pandas(desc="Computing")
data_path = pathlib.Path("/home/julien/Documents/all_signals_resampled/")

def extract_unique(a: xr.DataArray, dim: str):
    def get_uniq(a):
        nums = nunique(a, axis=-1, to_str=True)
        if (nums==1).all():
            r = np.take_along_axis(a, np.argmax(~pd.isna(a), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
            return r
        else:
            raise Exception(f"Can not extract unique value. Array:\n {a}\nExample\n{a[np.argmax(nums)]}")
    return xr.apply_ufunc(get_uniq, a, input_core_dims=[[dim]])


signals = xr.open_dataset("signals.nc").load()
metadata = xr.open_dataset("metadata.nc")


signals["time_representation_path"] = xr.apply_ufunc(lambda x: np.where(x=="", np.nan, x), signals["time_representation_path"])
signals = signals.where(signals["Structure"].isin(["GPe", "STN", "STR"]), drop=True)
group_index_cols = ["Species", "Structure", "Healthy"]
signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact'], coords=[signals["Contact"]])
signals = signals.set_coords("group_index")

grouped_results = xr.Dataset()
transfert_coords = ["CorticalState", "FullStructure", "Condition"]

for col in transfert_coords:
    grouped_results[col] = signals[col].groupby("group_index").map(extract_unique, dim="Contact").unstack()
grouped_results = auto_remove_dim(grouped_results)
grouped_results = grouped_results.set_coords(transfert_coords)

###### Now let's compute stuff on our signals
signals["bua_duration"] = apply_file_func(lambda arr: arr["index"].max() - arr["index"].min(), ".", signals["time_representation_path"].sel(sig_preprocessing="bua"), name="duration", save_group="./durations.pkl")
signals["spike_duration"] = apply_file_func(lambda arr: float(np.max(arr) - np.min(arr)), ".", signals["time_representation_path"].where(signals["sig_type"] == "spike_times"), name="spike_duration", save_group="./spike_durations.pkl")
signals["n_spikes"] = apply_file_func(lambda arr: arr.size, ".", signals["time_representation_path"].where(signals["sig_type"] == "spike_times"), name="n_spike", save_group="./n_spikes.pkl")
signals["n_spikes/s"] = signals["n_spikes"]/signals["spike_duration"]
signals["n_data_points"] = apply_file_func(lambda arr: float(arr.size), ".", signals["time_representation_path"], name="n_datapoints", save_group="./n_datapoints.pkl")
signals["_diff"] = (signals["bua_duration"] - signals["spike_duration"])

def compute_cwt(a, sig_type):
    import pywt
    if sig_type=="spike_times":
        fs=1000
        min = round(np.max(np.min(a)-0.002, 0), 3)
        max = round(np.max(a)+0.002, 3)
        arr = np.zeros(int((max-min)*fs) +1)
        np.add.at(arr, ((a - min)*fs).astype(int), 1)
        a=xr.DataArray(arr, dims="t", coords=[np.linspace(min, max, arr.size, endpoint=True)])
    else:
        test = a["t"].to_numpy()
        fs = np.diff(test)[0]
        if not (np.abs(np.diff(test)-fs) < 0.000001).all():
            print(fs, np.diff(test))
            err_index = np.argmax((np.diff(test)!=fs))
            raise Exception(f"Signal does not have a constant fs got {test[0:3]} and {test[err_index-1:err_index+1]}")
        else:
            fs=1/fs
    try:
        scales = pywt.frequency2scale('cmor1.5-1.0', np.arange(5, 50)/float(fs))
        coefs, _ = pywt.cwt(a, scales, 'cmor1.5-1.0')
    except:
        logger.error(f"Problem with fs={fs}")
        raise
    
    res= xr.DataArray(coefs, dims=["f", "t"], coords=[np.arange(5, 50), a["t"]])
    (s, e) = float(res["t"].min()), float(res["t"].max())
    res = res.groupby_bins("t", bins=np.linspace(s, e, int((e-s)*50))).mean()
    res = res.rename(t_bins="t")
    # print(res)
    # exit()
    return res

def test(a):
    from regular_index import RegularIndex
    # print(a)
    a = a.drop_indexes(["t"])
    a = a.set_xindex(["t"], RegularIndex)
    # a = a.reset_coords("t")
    print(a)
    # print(len(a.indexes))
    exit()

# print(signals.indexes)
# signals["test"] = apply_file_func(test, ".", signals["time_representation_path"])
signals["cwt_path"] = apply_file_func(compute_cwt, ".", signals["time_representation_path"], signals["sig_type"], out_folder="./cwt", name="cwt")
signals["time_freq_repr"] = apply_file_func(lambda a: np.abs(a * np.conj(a)), ".", signals["cwt_path"], out_folder="./tf_repr", name="time_freq_representation")
print(signals)
if not pathlib.Path("pair_signals.pkl").exists():
# if True:
    defined_signals = signals["cwt_path"].to_dataframe().reset_index()
    defined_signals = defined_signals.loc[~pd.isna(defined_signals["cwt_path"])]
    defined_signals = defined_signals.drop(columns="has_entry")
    defined_signals = defined_signals.drop(columns=["group_index"])
    pair_signals = toolbox.group_and_combine(defined_signals, ["Session", "FullStructure"], include_eq=False)
    pair_signals = pair_signals.loc[pair_signals["Contact_1"] != pair_signals["Contact_2"]]
    pair_signals["Contact_pair"] = pd.MultiIndex.from_arrays([pair_signals["Contact_1"], pair_signals["Contact_2"]], names=["Contact_1", "Contact_2"])
    pair_signals = pair_signals.set_index(["Contact_pair", "sig_preprocessing_1", "sig_preprocessing_2"])
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
    
    pair_signals = auto_remove_dim(pair_signals, ignored_vars=["Contact_pair"])
    pickle.dump(pair_signals, open("pair_signals.pkl", "wb"))
else:
    pair_signals = pickle.load(open("pair_signals.pkl", "rb"))
print(pair_signals)
pair_signals["has_entry_1"] = xr.apply_ufunc(lambda x: ~pd.isna(x), pair_signals["cwt_path_1"])
pair_signals["has_entry_2"] = xr.apply_ufunc(lambda x: ~pd.isna(x), pair_signals["cwt_path_2"])
pair_signals["group_index"] = xr.DataArray(pd.MultiIndex.from_arrays([pair_signals[a].data for a in group_index_cols],names=group_index_cols), dims=['Contact_pair'], coords=[pair_signals["Contact_pair"]])
 
pair_signals = pair_signals.set_coords([v for v in pair_signals.variables if not "cwt_path" in v])
pair_signals=pair_signals.where(((pair_signals["sig_type_1"] == "bua") & (pair_signals["sig_type_2"] == "spike_times")) | ((pair_signals["sig_type_2"] == "bua") & (pair_signals["sig_type_1"] == "spike_times")))
print(pair_signals)

def compute_coherence(a: xr.DataArray,b: xr.DataArray):
    a = a.to_dataset(name="a")
    b = b.to_dataset(name="b")
    print(a)
    print(b)
    a["t"] = xr.apply_ufunc(lambda x: np.round(x.left*50+0.00001)/50, a["t"], vectorize=True)
    b["t"] = xr.apply_ufunc(lambda x: np.round(x.left*50+0.00001)/50, b["t"], vectorize=True)
    tmp=xr.merge([a, b], join="inner")
    a = tmp["a"]
    b=tmp["b"]
    res = np.abs(a*np.conj(b))**2 / ((np.abs(a)**2)*(np.abs(b)**2))
    print(res.to_dataset(name="coherence"))
    input()
    return res

# a= 0.036004337535534746+0.018279389420020897j
# b = 98.23709725843953-769.0045991853731j
# print(np.abs(a)**2)
# print(np.abs(b)**2)
# print(np.abs(a*np.conj(b))**2)
# print((np.abs(a*np.conj(b))**2) / ((np.abs(a)**2)*(np.abs(b)**2)))
# exit()

# pair_signals["coherence"] = apply_file_func(compute_coherence, ".", pair_signals["cwt_path_1"], pair_signals["cwt_path_2"], out_folder="./coherence", name="coherence", n=2)
           
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

all_progress = tqdm.tqdm(desc="Extracting", total=float(len(signals["time_freq_repr"].groupby("group_index"))))
grouped_results["time_freq_repr_values"] = signals["time_freq_repr"].groupby("group_index").map(get_values).unstack()

grouped_results["pwelch"] = apply_file_func(lambda x: x.mean(), ".", grouped_results["time_freq_repr_values"], save_group="./pwelch.pkl", name="welch")

print(grouped_results)

welch = grouped_results["pwelch"].plot(x="f", hue="Species", style="Healthy", row="Structure", col="sig_preprocessing")
plt.show()



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
grouped_results = grouped_results.drop_dims("bins")
print(grouped_results.drop_vars(["CorticalState", "FullStructure", "Condition"]).to_dataframe().to_string())

# print(grouped_results.to_dataframe())

##### Now, let's plot the data

basic_data = grouped_results.drop_dims("bins").to_array("info", name="counts").to_dataframe().sort_index(level=["Species", "Structure", "Healthy"]).reset_index("Healthy", drop=True).set_index("Condition", append=True)
maxes=basic_data["counts"].groupby("info").max()
# basic_data["counts"] = basic_data["counts"]/maxes
basic_data["group"] = [str(x[1:]) for x in basic_data.index.values]

import seaborn as sns

g = sns.FacetGrid(data=basic_data.reset_index(), col="info", col_wrap=3, sharex=False, hue="Species", aspect=2).map_dataframe(sns.barplot, x="counts", y="group").tight_layout().add_legend()
for ax in g.axes.flat:
    ax.set_ylabel(None)
g.figure.subplots_adjust(top=.9, bottom=0.05)
# sns.barplot(data=basic_data, x = [str(x) for x in basic_data.index.values],  y="counts", hue="info", gap=0)
# plt.title(str(maxes))
plt.show()