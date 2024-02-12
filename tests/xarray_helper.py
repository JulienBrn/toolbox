import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger, shutil
from autosave import Autosave

logger=logging.getLogger(__name__)
class DimRemoveExcpt(Exception):pass

def apply_file_func(func, in_folder, path: xr.DataArray, *args, out_folder=None, name = None, recompute=False, save_group=None, n=1, path_arg=None, n_ret=1, output_core_dims=None, n_outcore_nans=None,**kwargs):
    def subapply(*args):
        # print("entering subapply")
        nonlocal nb_nans, nb_already_computed
        paths: List[pathlib.Path]=list(args[:n])
        args=args[n:]
        if (not pd.isna(paths).any()) and not ("nan" in paths):
            if not out_folder is None:
                dest: pathlib.Path = pathlib.Path(out_folder)
                progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans, status="Checking already computed"))
                for path in paths:
                    dest=dest/pathlib.Path(path).relative_to(pathlib.Path(in_folder))
                dest=dest.with_suffix(".pkl")
                # print(out_folder, dest, paths)
                if dest.exists() and not recompute:
                    nb_already_computed+=1
                    progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans))
                    progress.total=progress.total-1
                    progress.update(0)
                    return str(dest)
            data=[]
            progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans, status="Loading input data"))
            for path in paths:
                in_path: pathlib.Path  = pathlib.Path(in_folder)/path
                match in_path.suffix:
                    case  ".pkl":
                        data.append(pickle.load(in_path.open("rb")))
                    case ".npy":
                        data.append(toolbox.np_loader.load(in_path))
                    case ext:
                        raise Exception(f"Unknown extension {ext} for {in_path} from {path} {pd.isna(path)} {type(path)}")
            progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans, status="Computing"))
            # print(dest, dest.exists())
            if path_arg is None:
                ret = func(*data, *args)
            else:
                ret = func(*data, *args, **{path_arg:paths})
            progress.update(1)
            try:
                if isinstance(ret, xr.DataArray):
                    is_na = False
                else:
                    is_na = pd.isna(ret)
            except:
                is_na = False
            try:
                if is_na:
                    nb_nans+=1
                    progress.set_postfix(dict(saved=nb_already_computed, nb_nans=nb_nans))
            except:
                print(is_na)
                print(ret)
                raise
                # print("nan")
                # input()
            # print(f"Value considered {is_na}:\n{ret}")
            if not out_folder is None and not is_na:
                progress.set_postfix(dict(nb_nans=nb_nans, status="Dumping"))
                dest.parent.mkdir(exist_ok=True, parents=True)
                pickle.dump(ret, dest.with_suffix(".tmp").open("wb"))
                # print(out_folder, str(dest.with_suffix(".tmp")))
                # input()
                shutil.move(str(dest.with_suffix(".tmp")),str(dest))
                # print("Should exist !", dest, dest.exits())
                return str(dest)
            else:
                return ret
        else:
            # print("returning nan undefined input")
            
            if not output_core_dims is None:
                res = tuple(xr.DataArray(data=np.reshape([np.nan]*n_outcore_nans, [1]* (len(dims)-1)+[-1]), dims=dims) for dims in output_core_dims)
                # print(res)
                # input()

            else:
                res= tuple([np.nan for _ in range(n_ret)]) if not n_ret == 1 else np.nan
            return res
    if not save_group is None:
        group_path = pathlib.Path(save_group)
        if group_path.exists() and not recompute:
            return pickle.load(group_path.open("rb"))
    if float(path.count()) > 0:
        if name is None and not out_folder is None:
            progress = tqdm.tqdm(desc=f"Computing {out_folder}", total=float(path.count()))
        elif not name is None:
            progress = tqdm.tqdm(desc=f"Computing {name}", total=float(path.count()))
        else:
            progress = tqdm.tqdm(desc=f"Computing", total=float(path.count()))
    nb_nans=0
    nb_already_computed=0
    # print(args)
    # print(type(path))
    # print(path.ndim)
    progress.set_postfix(dict(status="Applying xarray ufunc"))
    res = xr.apply_ufunc(subapply, path, *args, vectorize=True, output_dtypes=None if out_folder is None else ([object]*n_ret), output_core_dims=output_core_dims if not output_core_dims is None else ((), ), **kwargs)
    if not save_group is None:
        group_path.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(res, group_path.with_suffix(".tmp").open("wb"))
        shutil.move(str(group_path.with_suffix(".tmp")),str(group_path))
    progress.update(0)
    return res

def nunique(a, axis, to_str=False):
            a = np.ma.masked_array(a, mask=pd.isna(a))
            if to_str:
                try:
                    a = a.astype(str)
                except Exception as e:
                    ta = a.reshape(-1)
                    for i in range(0, ta.size, 10):
                        try:
                            _ = ta[i:i+10].astype(str)
                        except:
                            e.add_note(f"Problem converting array values to string... Initial dtype is {a.dtype}. Example of values is\n{ta[i:i+10]} {ta[i+3]} {type(ta[i+3])}")
                            # print("Array is ", a)
                            raise e
                    e.add_note(f"Problem converting array values to string... Initial dtype is {a.dtype}. Example of values not found")
                    raise e
            sorted = np.ma.sort(a,axis=axis, endwith=True, fill_value=''.join([chr(255) for _ in range(5)]))
            unshifted = np.apply_along_axis(lambda x: x[:-1], axis, sorted)
            shifted = np.apply_along_axis(lambda x: x[1:], axis, sorted)
            diffs =  (unshifted != shifted)
            return np.ma.filled((diffs!=0).sum(axis=axis)+1, 1)


def auto_remove_dim(dataset:xr.Dataset, ignored_vars=None, kept_var=None, dim_list=None):
    def remove_numpy_dim(var: np.ndarray):
        nums = nunique(var, axis=-1, to_str=True)
        if (nums==1).all():
            return np.take_along_axis(var, np.argmax(~pd.isna(var), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
        else:
            raise DimRemoveExcpt("Can not remove dimension")
        
    ndataset = dataset
    if kept_var is None:
        vars = list(dataset.keys())+list(dataset.coords)
    else:
        vars = kept_var
    if not ignored_vars is None:
        vars = list(set(vars) - set(ignored_vars))
    # for var in set(list(dataset.keys())+list(dataset.coords)) - set(vars):
    #     ndataset[var] = dataset[var]
    for var in tqdm.tqdm(vars, desc="fit var dims", disable=True):
        # if var in ignored_vars:
        #     ndataset[var] = dataset[var]
        #     continue
        for dim in ndataset[var].dims:
            if dim_list is None or dim in dim_list:
                try:
                    ndataset[var] = xr.apply_ufunc(remove_numpy_dim, dataset[var], input_core_dims=[[dim]])
                except DimRemoveExcpt:
                    pass
                except Exception as e:
                    e.add_note(f"Problem while auto remove dim of variable={var}")
                    raise e
    return ndataset


def thread_vectorize(func, dim, max_workers=20, **kwargs):
    import concurrent
    new_args=[]


def resample_arr(a: xr.DataArray, dim: str, new_fs: float, position="centered", new_dim_name=None,*, mean_kwargs={}, return_counts=False):
    if new_dim_name is None:
        new_dim_name = f"{dim}_bins"
    match position:
        case "centered":
            a["new_fs_index"] = np.round(a[dim]*new_fs+1/(1000*new_fs))
        case "start":
            a["new_fs_index"] = (a[dim]*new_fs).astype(int)
    grp = a.groupby("new_fs_index")
    binned = grp.mean(dim, **mean_kwargs)
    binned = binned.rename(new_fs_index=new_dim_name)
    binned[new_dim_name] = binned[new_dim_name]/new_fs
    
    if return_counts:
        counts = grp.count(dim)
        counts= counts.rename(new_fs_index=new_dim_name)
        counts[new_dim_name] = counts[new_dim_name]/new_fs
        return binned, counts
    else:
        return binned
    
def sampled_arr_from_events(a: np.array, fs: float, weights=1):
    if len(a.shape) > 1:
        raise Exception(f"Wrong input shape. Got {a.shape}")
    m=np.round(np.min(a)*fs)
    M=np.round(np.max(a)*fs)
    n = int(M-m + 1)
    res = np.zeros(n)
    np.add.at(res, np.round(a*fs).astype(int) - int(m), weights)
    return res, m/fs

def sum_shifted(a: np.array, kernel: np.array):
    if len(a.shape) > 1:
        raise Exception(f"Wrong input shape. Got {a.shape}")
    if len(kernel.shape) > 1:
        raise Exception(f"Wrong input shape. Got {kernel.shape}")
    if kernel.size % 2 !=1:
        raise Exception(f"Kernel must have odd size {kernel.size}")
    roll = int(np.floor(kernel.size/2))
    a = np.concatenate([np.zeros(roll), a, np.zeros(roll)])
    # kernel = np.concatenate([kernel, np.zeros(a.size - kernel.size)])
    res = np.zeros(a.size)
    for i in range(-roll, roll+1):
        res = res + np.roll(a,i)*kernel[i+roll]
    return res

def normalize(a: xr.DataArray):
    std = a.std()
    a_normal = (a - a.mean()) / std
    return a_normal

def apply_file_func_decorator(base_folder, **kwargs):
    def decorator(f):
        def new_f(*arr_paths):
            return apply_file_func(f, base_folder, *arr_paths, **kwargs)
        return new_f
    return decorator

def extract_unique(a: xr.DataArray, dim: str):
    def get_uniq(a):
        nums = nunique(a, axis=-1, to_str=True)
        if (nums==1).all():
            r = np.take_along_axis(a, np.argmax(~pd.isna(a), axis=-1, keepdims=True), axis=-1).squeeze(axis=-1)
            return r
        else:
            raise Exception(f"Can not extract unique value. Array:\n {a}\nExample\n{a[np.argmax(nums)]}")
    return xr.apply_ufunc(get_uniq, a, input_core_dims=[[dim]])


def mk_bins(a: xr.DataArray, dim, new_dim, coords, weights=None):
    # print(f"input arr sum {float(a.sum())}")
    if not weights is None:
        tmp = xr.apply_ufunc(lambda x, y: f"{x}_{y}", a, weights, vectorize=True)
    else:
        tmp = xr.apply_ufunc(lambda x: f"{x}_{1}", a, vectorize=True)
    weights = None
    # print(a)
    # print(weights)
    # exit()
    def compute(a: np.ndarray, weights=None):
        # if np.nansum(a) >0:
        #     print(f"compute input sum {np.nansum(a)}")
        # print(a)
        def make_hist(a: np.ndarray, weights=None):
            # print(f"make_hist input sum {np.nansum(a)}")
            # import time
            # time.sleep(0.01)
            tmp = np.array([list(x.split("_")) for x in a])
            a,weights = tmp[:, 0].astype(float), tmp[:, 1].astype(float)
            h, edges = np.histogram(a, coords, weights=weights)
            return h
        if weights is None:
            r = np.apply_along_axis(make_hist, axis=-1, arr= a)
        else:
            r = np.apply_along_multiple_axis(make_hist, axis=-1, arrs= [a, weights])
        # print(r)
        return r
    if weights is None:
        res: xr.DataArray = xr.apply_ufunc(compute, tmp, input_core_dims=[[dim]], output_core_dims=[[new_dim]])
    else:
        res: xr.DataArray = xr.apply_ufunc(compute, a, weights, input_core_dims=[[dim], [dim]], output_core_dims=[[new_dim]])
    # print(res)
    res = res.assign_coords({new_dim:(coords[1:]+ coords[:-1])/2, f"{new_dim}_low_edge": (new_dim, coords[:-1]), f"{new_dim}_high_edge": (new_dim, coords[1:])})
    return res