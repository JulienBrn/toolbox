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
signals: pd.DataFrame = toolbox.df_loader.load(data_path/"database.tsv")
metadata: pd.DataFrame = toolbox.df_loader.load(data_path/"metadata.tsv")
signals = signals.merge(metadata)
signals = signals.loc[signals["signal_resampled_type"]!="spike_bins"].copy()
spikes_metadata = toolbox.df_loader.load(data_path/"spikes_data_metadata.tsv")
spikes_metadata.drop(columns=['diff_duration_error', 'spike_bins_fs', "min_nb", "spike_bins_signal"], inplace=True)
spikes_metadata["signal_resampled_type"] = "spike_times"
spikes_metadata["signal_resampled_fs"] = spikes_metadata["input_signal_fs"]
spikes_data = toolbox.df_loader.load(data_path/"spikes_database.tsv")
spikes_data = spikes_data.merge(spikes_metadata)
spikes_data["signal_resampled_fs"] = np.where(spikes_data["signal_resampled_fs"].str.contains("Rec"), 48076.92337036133, spikes_data["signal_resampled_fs"]).astype(float)

# print(spikes_data[["Species", "signal_resampled_fs", "Date"]])
# print(set(signals.columns.to_list()) ^  set(spikes_data.columns.to_list()))
# print(set(signals.columns.to_list()) -  set(spikes_data.columns.to_list()))
# exit()
signals = pd.concat([signals, spikes_data])

signals["CorticalState"] = signals["Species"].apply(lambda x: "Anesthetized" if x=="Rat" else "Awake" if x=="Human" else "Awake" if x=="Monkey" else "Unknown")
signals["Condition"] = np.where(signals["Healthy"] ==True, "Control",
                       np.where(signals["Species"] =="Monkey", "mptp",
                       np.where(signals["Species"] =="Rat", "6ohda",
                       np.where(signals["Species"] =="Human", "Park", 
                        "Unknown" ))))
signals["Structure"] = np.where(signals["Structure"] =="MSN", "STR", signals["Structure"])
signals["FullStructure"] = signals["Structure"]
signals["Structure"] = np.where(signals["Structure"] =="STN_DLOR", "STN",
                       np.where(signals["Structure"] =="STN_VMNR", "STN",
                       signals["Structure"]))

signals["Healthy"] = np.where(signals["FullStructure"] =="STN_VMNR", True, signals["Healthy"])
signals["Condition"] = np.where(signals["FullStructure"] =="STN_VMNR", "Control_vnmr", signals["Condition"])

signals["Date"] = signals["Date"].str.replace("_", "")
signals["Date"] = np.where(signals["Date"].str.len()<8,"20"+ signals["Date"].str.slice(0, 6), signals["Date"])
signals["Date"] = pd.to_datetime( signals["Date"], format="%Y%m%d", exact=True)

signals["Duration"] = pd.to_numeric(signals["Duration"], errors="coerce")

def get_set(x:str):
    import ast
    try:
        r = ast.literal_eval(x)
    except:
        return x
    else:
        if isinstance(r, set):
            if len(r)==1:
                return str(list(r)[0])
            else:
                return None
        elif isinstance(r, int) or isinstance(r, float):
            return x
        else: raise Exception(f"Strange got {r} from {x}")

tot=0
def compute_id(d: pd.DataFrame, col: str, new_col):
    global tot
    tot = tot + len(d.index)
    r = d.groupby(col, dropna=False).ngroup()
    res = d.copy()
    res[new_col] = r
    return res

signals["Contact"]: pd.Series = np.where(signals["Species"] =="Rat", "E:"+signals["Electrode"].apply(get_set).astype(str) + "C:"+signals["Probes"].apply(get_set),
                     np.where(signals["Species"] =="Monkey", "E:"+signals["Electrode"].astype(str) +"D:"+ signals["Depth"].astype(str),
                     np.where(signals["Species"] =="Human", "E:"+signals["Electrode"].astype(str) +"D:" + signals["Depth"].astype(str) +"H:"+ signals["Hemisphere"].astype(str),
                     "Unknown" )))


invalid_contacts = signals["Contact"].isna()
logger.info(f"Removing {invalid_contacts.sum()} signals due to invalid contacts")

signals=signals.loc[~invalid_contacts, :]

tqdm.tqdm.pandas(desc="Making contact/sig_preprocessing dimensions")
signals["Subject"] = signals["Subject"].fillna("Unknown")
signals["Contact"] = signals.groupby(["Species", "Subject","Session","Structure", "Contact"]).ngroup()
signals: pd.DataFrame = signals.groupby("Contact", dropna=False).progress_apply(lambda d: compute_id(d, "Unit", "unit_id")).reset_index(drop=True)
signals["sig_preprocessing"] = np.where(signals["signal_resampled_type"]=="spike_times", "neuron_" + signals["unit_id"].astype(str),signals["signal_resampled_type"])    


signals.rename(columns=dict(signal_resampled_type="sig_type", input_signal_path="signal_file_source", path="time_representation_path", signal_resampled_fs="sig_fs"), inplace=True)

signals.set_index(["Contact", "sig_preprocessing"], inplace=True)

# 
signals: pd.DataFrame
if len(signals.loc[signals.index.duplicated(keep=False)].index) != 0:
    print(f'Duplicated indices...\n{signals.loc[signals.index.duplicated(keep=False), ["Species","Subject", "sig_type", "signal"]].sort_index().to_string()}')
    exit()


signals: xr.Dataset = xr.Dataset.from_dataframe(signals)
signals = signals.drop_vars(["column", "signal_resampled", "input_signal_type", "input_signal", "input_signal_fs", 'nb_units_discarded', "End", "max_spike_time", "unit_id"])

auto_remove_dim(signals)


signals = signals.set_coords(["Species", "FullStructure", "Structure", "Condition", "Subject", "Session", "Date", "CorticalState", "Healthy", "sig_type",])
# signals["raw_fs"] = signals["input_signal_fs"].sel(sig_preprocessing="bua")
# signals["spike_fs"] = signals["sig_fs"].sel(sig_preprocessing="neuron_0")
# signals["pathn0"] = signals["time_representation_path"].sel(sig_preprocessing="neuron_0")
# print(signals[["raw_fs", "spike_fs", "pathn0"]].to_dataframe())

def mk_sig_time_representation(arr, start, fs, sig_type):
    if not isinstance(arr, np.ndarray):
        return np.nan
    if sig_type == "spike_times":
        ret = arr/fs + start
        return ret
    else:
        # s = pd.Series(arr, index=np.arange(arr.size) / fs + start)
        # ret = xr.Dataset()
        # ret["amplitude"] = xr.DataArray(arr, dims=["t"], coords=[np.arange(arr.size) / fs + start])
        # ret["fs"] = fs
        # print(ret)
        # input()
        # return ret
        return xr.DataArray(arr, dims=["t"], coords=[np.arange(arr.size) / fs + start])

signals["time_representation_path"] = apply_file_func(mk_sig_time_representation, data_path, signals["time_representation_path"], signals["Start"], signals["sig_fs"], signals["sig_type"], out_folder = "./time_repr", name="time_repr")
signals["has_entry"] = xr.apply_ufunc(lambda x: ~pd.isna(x), signals["time_representation_path"])
signals = signals.set_coords("has_entry")
signals["has_data_but_no_bua"] = signals["has_entry"].max(dim="sig_preprocessing") &  (~signals["has_entry"].sel(sig_preprocessing="bua"))
logger.info(f"Removing {float(signals['has_data_but_no_bua'].sum())} contacts signals because they had no raw signal...")
# print(signals.where(signals["has_data_but_no_bua"], drop=True).to_csv("tmp.csv"))
signals = signals.where(~signals["has_data_but_no_bua"], drop=True)
signals = signals.drop_vars("has_data_but_no_bua")
metadata = signals.drop_vars(["time_representation_path"])
signals = signals[["time_representation_path"]]
print(metadata)
print(signals)
signals.to_netcdf("signals.nc")
metadata.to_netcdf("metadata.nc")

# 
exit()




max = signals["has_entry"].where(signals["Species"]=="Rat").groupby("Session").sum().where(signals["sig_type"]=="spike_bins").sum("sig_preprocessing").max()
idxmax = signals["has_entry"].where(signals["Species"]=="Rat").groupby("Session").sum().where(signals["sig_type"]=="spike_bins").sum("sig_preprocessing").idxmax()
session_data = xr.Dataset()
session_data["Max_n_neurons"] = max
session_data["Max_n_neurons_session"] = idxmax
print(session_data)
signals = signals[["time_representation_path", "has_entry", "sig_fs", "Start"]]



signals["Duration"] = apply_file_func(lambda arr, fs: arr.size/fs if isinstance(arr, np.ndarray) else np.nan, data_path, signals["time_representation_path"], signals["sig_fs"], name="duration")

signals.to_netcdf("signals.nc")
print(signals)
exit()

progress = tqdm.tqdm(desc="Computing", total=signals.sizes["index"])

def mk_time_df(path, fs, start):
    progress.update(1)
    data = toolbox.np_loader.load(data_path/path)
    r = pd.Series(data)
    r.name="data"
    r = r.to_frame()
    r["t"] = r.index*fs +start
    r = r.set_index("t").squeeze()
    return r

signals["time_representation"] = xr.apply_ufunc(mk_time_df, signals["time_representation_path"], signals["signal_resampled_fs"], signals["Start"], vectorize=True)
# input(signals)
# signals["signal"] = signals["signal"].groupby("sig_id", restore_coord_dims=False).apply(lambda grp: xr.apply_ufunc(lambda x: x.data[0], grp, input_core_dims=[grp.dims]))
# signals["Healthy"] = signals["Healthy"].groupby("Condition", restore_coord_dims=False).apply(lambda grp: xr.apply_ufunc(lambda x: x.data[0], grp, input_core_dims=[grp.dims]))
input(signals)




def embed(path, fs, start):
    import sparse
    new_data = np.array([dict(data_file=path, fs=fs) for path, fs in zip(path.data, fs.data)])
    for s1, s2 in zip([path, fs, start], [path, fs, start][1:]):
        if not np.equal(s1.coords, s2.coords).all():
            raise Exception("Incompatible data")
    return  sparse.COO(coords = path.coords, data=new_data, fill_value=np.nan)

signals["time_data"] = xr.apply_ufunc(
    embed, 
    signals["path"], signals["signal_resampled_fs"], signals["Start"])

signals = signals.drop_vars(["signal_resampled_fs", "Start", "path"])

input(signals)
# input(list(signals.keys()))
input(signals["Healthy"])