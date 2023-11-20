import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from autosave import Autosave

class DimRemoveExcpt(Exception):pass

def apply_file_func(func, in_folder, path: xr.DataArray, *args, out_folder=None, name = None, recompute=False, save_group=None):
    
    def subapply(path, *args):
        if not pd.isna(path) and not path == "":
            if not out_folder is None:
                dest: pathlib.Path = pathlib.Path(out_folder)/path
                dest=dest.with_suffix(".pkl")
                if dest.exists() and not recompute:
                    return str(dest)
            in_path: pathlib.Path  = pathlib.Path(in_folder)/path
            match in_path.suffix:
                case  ".pkl":
                    data = pickle.load(in_path.open("rb"))
                case ".npy":
                    data = toolbox.np_loader.load(in_path)
                case ext:
                    raise Exception(f"Unknown extension {ext} for {in_path} from {path}")
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
    if not save_group is None:
        group_path = pathlib.Path(save_group)
        if group_path.exists() and not recompute:
            return pickle.load(group_path.open("rb"))
    if name is None and not out_folder is None:
        progress = tqdm.tqdm(desc=f"Computing {out_folder}", total=float(path.count()))
    elif not name is None:
        progress = tqdm.tqdm(desc=f"Computing {name}", total=float(path.count()))
    else:
        progress = tqdm.tqdm(desc=f"Computing", total=float(path.count()))
    res = xr.apply_ufunc(subapply, path, *args, vectorize=True, output_dtypes=None if out_folder is None else [object])
    if not save_group is None:
        pickle.dump(res, group_path.open("wb"))
    return res

def nunique(a, axis, to_str=False):
            a = np.ma.masked_array(a, mask=pd.isna(a))
            if to_str:
                a = a.astype(str)
            sorted = np.ma.sort(a,axis=axis, endwith=True, fill_value=''.join([chr(255) for _ in range(5)]))
            unshifted = np.apply_along_axis(lambda x: x[:-1], axis, sorted)
            shifted = np.apply_along_axis(lambda x: x[1:], axis, sorted)
            diffs =  (unshifted != shifted)
            return np.ma.filled((diffs!=0).sum(axis=axis)+1, 1)


def auto_remove_dim(dataset:xr.Dataset, ignored_vars=[]):
    def remove_numpy_dim(var: np.ndarray):
        nums = nunique(var, axis=-1, to_str=True)
        if (nums==1).all():
            return np.take_along_axis(var, np.argmax(~pd.isna(var), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
        else:
            raise DimRemoveExcpt("Can not remove dimension")
        
    ndataset = dataset
    for var in tqdm.tqdm(list(dataset.keys())+list(dataset.coords), desc="fit var dims"):
        if var in ignored_vars:
            ndataset[var] = dataset[var]
            continue
        for dim in ndataset[var].dims:
            try:
                ndataset[var] = xr.apply_ufunc(remove_numpy_dim, dataset[var], input_core_dims=[[dim]])
            except DimRemoveExcpt:
                pass
            except Exception as e:
                e.add_note(f"Problem while auto remove dim of variable={var}")
                raise e
    return ndataset