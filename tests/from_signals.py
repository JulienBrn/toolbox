import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave

class DimRemoveExcpt(Exception):pass

logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
tqdm.tqdm.pandas(desc="Computing")


data_path = pathlib.Path("/home/julien/Documents/all_signals_resampled/")
signals: pd.DataFrame = toolbox.df_loader.load(data_path/"database.tsv")
metadata: pd.DataFrame = toolbox.df_loader.load(data_path/"metadata.tsv")
signals = signals.merge(metadata)
print(signals)
signals["CorticalState"] = signals["Species"].apply(lambda x: "Anesthetized" if x=="Rat" else "Awake" if x=="Human" else "Awake" if x=="Monkey" else "Unknown")
signals["Condition"] = np.where(signals["Healthy"] ==True, "Control",
                       np.where(signals["Species"] =="Monkey", "mptp",
                       np.where(signals["Species"] =="Rat", "6ohda",
                       np.where(signals["Species"] =="Human", "Park", 
                        "Unknown" ))))

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

signals["Contact"]: pd.Series = np.where(signals["Species"] =="Rat", "E:"+signals["Electrode"].astype(str) + "C:"+signals["Probes"].progress_apply(get_set),
                     np.where(signals["Species"] =="Monkey", "E:"+signals["Electrode"].astype(str) +"D:"+ signals["Depth"].astype(str),
                     np.where(signals["Species"] =="Human", "E:"+signals["Electrode"].astype(str) +"D:" + signals["Depth"].astype(str) +"H:"+ signals["Hemisphere"].astype(str),
                     "Unknown" )))

# input(signals["Contact"])
# input(signals["Contact"].isnull())

invalid_contacts = signals["Contact"].isna()
# input(signals.loc[invalid_contacts, ["Structure", "Unit", "Probes", "Electrode", "Session"]].to_string())
logger.info(f"Removing {invalid_contacts.sum()} signals due to invalid contacts")

signals=signals.loc[~invalid_contacts, :]

signals["Subject"] = signals["Subject"].fillna("Unknown")
signals["Contact"] = signals.groupby(["Species", "Subject","Session","Structure", "Contact"]).ngroup()
signals: pd.DataFrame = signals.groupby("Contact", dropna=False).progress_apply(lambda d: compute_id(d, "Unit", "unit_id")).reset_index(drop=True)
signals["sig_preprocessing"] = np.where(signals["signal_resampled_type"]=="spike_bins", "spike " + signals["unit_id"].astype(str),signals["signal_resampled_type"])    




 
# signals["Session"] = signals.groupby(["Species"]).progress_apply(lambda d: compute_id(d, "Session", "species_session_num")).reset_index(drop=True)
# signals = signals.groupby(["Species", "Structure", "CorticalState", "Condition"]).progress_apply(lambda d: compute_id(d, "Session", "grp_session_num")).reset_index(drop=True)
# input(signals)
# signals = signals.groupby(["Species", "species_session_num"]).progress_apply(lambda d: compute_id(d, "Probes", "session_probe_num")).reset_index(drop=True)
# signals: pd.DataFrame = signals.groupby(["Species", "Structure", "CorticalState", "Condition", "species_session_num", "session_signal_num"]).progress_apply(lambda d: compute_id(d, "Unit", "unit_id")).reset_index(drop=True)
# signals["sig_preprocessing"] = np.where(signals["signal_resampled_type"]=="spike_bins", "spike " + signals["unit_id"].astype(str),signals["signal_resampled_type"])



signals.rename(columns=dict(signal_resampled_type="sig_type", input_signal_path="signal_file_source", path="time_representation_path", signal_resampled_fs="sig_fs"), inplace=True)

signals.set_index(["Contact", "sig_preprocessing"], inplace=True)

# 
signals: pd.DataFrame
if len(signals.loc[signals.index.duplicated(keep=False)].index) != 0:
    print(f'Duplicated indices...\n{signals.loc[signals.index.duplicated(keep=False), ["Species","Subject", "sig_type", "signal"]].sort_index().to_string()}')
    exit()

# input(signals)
signals: xr.Dataset = xr.Dataset.from_dataframe(signals)
signals = signals.drop_vars(["column", "signal_resampled", "input_signal_type", "input_signal", "input_signal_fs", 'nb_units_discarded', "End", "Duration", "Contact", "max_spike_time", "unit_id"])
# input(signals)

def auto_remove_dim(dataset:xr.Dataset):
    def remove_numpy_dim(var: np.ndarray):
        def nunique(a, axis, to_str=False):
            a = np.ma.masked_array(a, mask=pd.isna(a))
            if to_str:
                a = a.astype(str)
            sorted = np.ma.sort(a,axis=axis, endwith=True, fill_value=''.join([chr(255) for _ in range(5)]))
            unshifted = np.apply_along_axis(lambda x: x[:-1], axis, sorted)
            shifted = np.apply_along_axis(lambda x: x[1:], axis, sorted)
            diffs =  (unshifted != shifted)
            return np.ma.filled((diffs!=0).sum(axis=axis)+1, 1)

        nums = nunique(var, axis=-1, to_str=True)
        if (nums==1).all():
            return np.take_along_axis(var, np.argmax(~pd.isna(var), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
        else:
            raise DimRemoveExcpt("Can not remove dimension")
        
    ndataset = dataset
    for var in dataset:
        logger.info(f"Handling variable {var}")
        try:
            ndataset[var] = xr.apply_ufunc(remove_numpy_dim, dataset[var], input_core_dims=[["sig_preprocessing"]])
        except DimRemoveExcpt:
            pass
        try:
            ndataset[var] = xr.apply_ufunc(remove_numpy_dim, dataset[var], input_core_dims=[["Contact"]])
        except DimRemoveExcpt:
            pass
    return ndataset

auto_remove_dim(signals)


signals = signals.set_coords(["Species", "Structure", "Condition", "Subject", "Session", "Date", "CorticalState", "Healthy", "sig_type", "signal_file_source",])

# input(signals)

signals["has_entry"] = xr.apply_ufunc(lambda x: ~pd.isna(x), signals["time_representation_path"])
max = signals["has_entry"].where(signals["Species"]=="Rat").groupby("Session").sum().where(signals["sig_type"]=="spike_bins").sum("sig_preprocessing").max()
idxmax = signals["has_entry"].where(signals["Species"]=="Rat").groupby("Session").sum().where(signals["sig_type"]=="spike_bins").sum("sig_preprocessing").idxmax()
session_data = xr.Dataset()
session_data["Max_n_neurons"] = max
session_data["Max_n_neurons_session"] = idxmax
print(session_data)
signals = signals[["time_representation_path", "has_entry", "sig_fs", "Start"]]

def apply_file_func(func, in_folder, path: xr.DataArray, *args, out_folder=None, name = None, recompute=False):
    if name is None and not out_folder is None:
        progress = tqdm.tqdm(desc=f"Computing {out_folder}", total=float(path.count()))
    elif not name is None:
        progress = tqdm.tqdm(desc=f"Computing {name}", total=float(path.count()))
    else:
        progress = tqdm.tqdm(desc=f"Computing", total=float(path.count()))
    def subapply(path, *args):
        if not pd.isna(path):
            if not out_folder is None and not recompute:
                dest: pathlib.Path = pathlib.Path(out_folder)/path
                if dest.exists():
                    return str(dest)
            in_path: pathlib.Path  = pathlib.Path(in_folder)/path
            match in_path.suffix:
                case  ".pkl":
                    data = pickle.load(in_path.open("rb"))
                case ".npy":
                    data = toolbox.np_loader.load(in_path)
                case _:
                    raise Exception("Unknown extension")
            ret = func(data, *args)
            progress.update(1)
            if not out_folder is None and not np.isnan(ret).all():
                dest.parent.mkdir(exist_ok=True, parents=True)
                pickle.dump(ret, dest.open("wb"))
                return str(dest)
            else:
                return ret
        else:
            return path
    return xr.apply_ufunc(subapply, path, *args, vectorize=True)

signals["Duration"] = apply_file_func(lambda arr, fs: arr.size/fs if isinstance(arr, np.ndarray) else np.nan, data_path, signals["time_representation_path"], signals["sig_fs"], name="duration")
def mk_sig_time_representation(arr, start, fs):
    if not isinstance(arr, np.ndarray):
        return np.nan
    s = pd.Series(arr, index=np.arange(arr.size) / fs + start)
    return xr.DataArray.from_series(s)

signals["time_representation_path"] = apply_file_func(mk_sig_time_representation, data_path, signals["time_representation_path"], signals["Start"], signals["sig_fs"], out_folder = "./time_repr", name="time_repr")
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