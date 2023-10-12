from __future__ import annotations
from typing import List, Dict, Any, Set
import xarray as xr, numpy as np, logging, pandas as pd

logger = logging.getLogger(__name__)

class DataArray:
    a: xr.DataArray
    level_name: str
    subdims: Set[str]
    subcoords: Set[str]
    name: str

    def __init__(self):pass
    def copy(self):
        r = DataArray()
        r.a = self.a.copy()
        r.level_name = self.level_name
        r.subdims = self.subdims.copy()
        r.subcoords = self.subcoords.copy()
        r.name = self.name
        return r

    @staticmethod
    def from_xarray(a: xr.DataArray) -> DataArray:
        r = DataArray()
        r.a = a
        r.subdims = {}
        r.level_name = None
        r.subcoords = {}
        r.name = a.name
        # input(f"fxarray {r.name}")
        return r
    
    def from_xarray_level(a: xr.DataArray, level_name: str) -> DataArray:
        xr.DataArray().sel()
        r = DataArray()
        r.a = a
        r.level_name = level_name
        if a.size == 0:
            r.subdims =set({})
            r.subcoords =set({})
            r.name = "empty dataarray"
        else:
            # print(a)
            subdims = xr.apply_ufunc(lambda x: set(x.a.dims).union(x.subdims) if isinstance(x, DataArray) else {None}, a, vectorize=True).values.flat
            subcoords: np.ndarray = xr.apply_ufunc(lambda x: set(x.a.coords).union(x.subcoords) if isinstance(x, DataArray) else {None}, a, vectorize=True).values.flat
            names = xr.apply_ufunc(lambda x: x.name if isinstance(x, DataArray) else {None}, a, vectorize=True).values.flat
            # print(subcoords)
            s = subdims[0]
            # print(s)
            if ((subdims == s) | (subdims=={None})).all():
                r.subdims = s
            else:
                raise ValueError(f"Different subdims. Examples:\n{subdims[subdims != s]}")
            
            subcoords_set = [x for x in subcoords if not x =={None}]
            r.subcoords = set.intersection(*subcoords_set) if subcoords_set else set() 
            r.name = names[0]
            # input(f"r={r.name}")
        return r
    
    def __str__(self):
        s = f"""
Current Level: {self.level_name}
Dimensions: {', '.join([f'{d}({s})' for d,s in self.a.sizes.items()])}, {', '.join(self.subdims)}
{self.a.coords}
Embedded coordinates: {', '.join(self.subcoords)}
Example Values (2 max per dimension)
{self.to_series(n=2)}""".strip()
        return s
    
    def sel(self, indexers: Dict[str, Any]=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        if indexers is None:
            indexers = indexers_kwargs
        cur_indexers = {k:v for k,v in indexers.items() if k in self.a.dims}
        other_indexers = {k:v for k,v in indexers.items() if not k in self.a.dims}
        res = self.copy()
        # print(f"Level {self.level_name}")
        # print(self.a)
        def compute_embedded_sel(x):
            if isinstance(x, DataArray):
                return x.sel(other_indexers, method, tolerance, drop)
            else:
                return np.nan
        if other_indexers == {}:
            res.a = self.a.sel(cur_indexers, method, tolerance, drop)
        elif cur_indexers== {}:
            res.a = xr.apply_ufunc(compute_embedded_sel, self.a, vectorize=True)
        else:
            res.a = xr.apply_ufunc(compute_embedded_sel, self.a.sel(cur_indexers, method, tolerance, drop), vectorize=True)
        return res
    
    def _stack_dims_into_lower_level(self, dims: Set[str]):
        ndimname = "_tmp_stack"
        curr_dims = {x for x in dims if x in self.a.dims}
        other_dims = {x for x in dims if not x in self.a.dims}
        if self.level_name is None:
            if len(curr_dims) == 0:
                return self
            else:
                res=self.copy()
                def f(dn: np.ndarray):
                    dn = dn.reshape(list(dn.shape[:-len(dims)]) +[-1])
                    return dn
                res.a = xr.apply_ufunc(f, res.a, input_core_dims=[curr_dims], output_core_dims=[ndimname])
                return res
        else:
            if len(curr_dims) == 0:
                res=self.copy()
                res.a = xr.apply_ufunc(lambda x: x._stack_dims_into_lower_level(other_dims), res.a, vectorize=True)
                return res
            else:
                res = self.copy()
                def f2(dn: np.ndarray):
                    dn = dn.reshape(list(dn.shape[:-len(dims)]) +[-1])
                    def agg(d):
                        data = {i:v.a for i,v in enumerate(d) if isinstance(v, DataArray)}
                        xr.apply_ufunc(f3, *data.values(), input_core_dims=[[other]])
                    return np.apply_along_axis(agg, len(dn.shape)-1, dn)
                res.a = xr.apply_ufunc(f2, res.a, input_core_dims=[curr_dims], output_core_dims=[ndimname])

    def _push_dims_to_one_level_down(self, dims: Set[str]):
        ndimname = "_tmp"
        if len(self.subdims) ==0:
            return self
        def agg_func(dn: np.ndarray):
            dn = dn.reshape(list(dn.shape[:-len(dims)]) +[-1])
            # logger.info(f"entered {dn.shape}")
            def agg(d):
                # logger.info(f"agg {d.shape}")
                data = {i:v.a for i,v in enumerate(d) if isinstance(v, DataArray)}
                if len(data) > 0:
                    r = xr.Dataset(data).to_array(dim=ndimname)
                    # print(r)
                    if isinstance(r.values.flat[0], DataArray):
                        return DataArray.from_xarray_level(r, str(d[0].level_name))
                    else:
                        return DataArray.from_xarray(r)
                else:
                    return np.nan
            return np.apply_along_axis(agg, len(dn.shape)-1, dn)
        res = self.copy()
        res.a = xr.apply_ufunc(agg_func, self.a, input_core_dims=[dims])
        # if res.a.count() < 2 and squeeze_levels:
        #     # print(res)
        #     res = res.a.values.flat[0]
        #     # input(res)
        # else:
        res.subdims = res.subdims.union(dims)
        return res
    
    def get_shape_sizes_dict(self):
        d = {f"curr({self.level_name})": [dict(self.a.sizes)]}
        if len(self.subdims) > 0:
            d["next"] = [x.get_shape_sizes_dict() for x in self.a.values.flat[0:4] if isinstance(x, DataArray)]
        return d
    
    def to_series(self, n=None):
        import pandas as pd
        self.a.name = self.name if self.name else "unnamed"
        if not n is None:
            d = self.a.isel({d:slice(0,n) for d in self.a.dims})
              
        else:
            d = self.a
        if d.size ==1:
            if not isinstance(d.values.flat[0], DataArray):
                return d.values.flat[0]
            if len(d.coords) == 0: 
                return d.values.flat[0].to_series(n)
            # input(d)
            # input({k:v.values.flat[0] for k,v in dict(d.coords).items()})
            # d = d.values.flat[0]
            ind = [v.values.flat[0] for v in d.coords.values()]
            df = pd.DataFrame([ind+[d.values.flat[0]]], columns=list(d.coords.keys())+[self.a.name])
            df=d.set_index(list(d.coords.keys()))
            # input(d)
            # input(type(d))
        else:
            df = d.to_dataframe()
        df = df[self.a.name]

        if self.level_name is None:
            return df
        res = df.groupby(by = df.index.names, observed=True).apply(lambda x: None if not isinstance(x.iat[0], DataArray) else x.iat[0].to_series(n=n))
        return res

    def _compute_combine(self, dims: Set[str], leaf_func, gather_func, squeeze_levels=True, fst=True):
        logger.info(f"STEP({self.level_name}, {dims}, {fst}) Entered") 
        curr_dims = {x for x in dims if x in self.a.dims}
        other_dims = {x for x in dims if not x in self.a.dims}
        if self.level_name is None:
            # if len(curr_dims) > 0:
            res = self.copy()
            logger.info(f"STEP({self.level_name}, {dims}) Applying leaf func over {curr_dims}")
            if fst:
                res.a = leaf_func(res.a, curr_dims)
                # print(res.a)
                # input()
            else:
                res.a = gather_func(res.a, curr_dims)
            if "_tmp" in res.a.dims:
                logger.error("_tmp")
                exit()
            return res
            # else:
            #     logger.info(f"STEP({self.level_name}, {dims}) Returning self")
            #     if "_tmp" in self.a.dims:
            #         logger.error("_tmp")
            #         exit()
            #     return self
        else:
            res = self.copy()
            logger.info(f"STEP({self.level_name}, {dims}) Applying recursively (other_dims)")
            res.a = xr.apply_ufunc(lambda x: x._compute_combine(other_dims, leaf_func, gather_func, fst=True) if isinstance(x, DataArray) else np.nan, self.a, vectorize=True)
            if len(curr_dims) > 0:
                
                # print(res.get_shape_sizes_dict())
                logger.info(f"    {res.a.values.flat[0]}")
                logger.info(f"STEP({self.level_name}, {dims}) Pushing dims {curr_dims} down")
                res = res._push_dims_to_one_level_down(curr_dims)
                if "_tmp" in res.a.dims:
                    logger.error(f"_tmp {self.level_name}, {dims}")
                    exit()
                logger.info(f"    {res.a.values.flat[0]}")
                # print(res.get_shape_sizes_dict())
                # input()
                logger.info(f"STEP({self.level_name}, {dims})  Applying recursively (_tmp)")
                try:
                    res.a = xr.apply_ufunc(lambda x: x._compute_combine({"_tmp"}, leaf_func, gather_func, fst=False) if isinstance(x, DataArray) else np.nan, res.a, vectorize=True)
                    if squeeze_levels and res.a.count() <2:
                        res=res.a.values.flat[0]
                except:
                    logger.error(f"STEP({self.level_name}, {dims}) Exception\n{res.a}")
                    raise
                    exit()
                logger.info(f"STEP({self.level_name}, {dims}) Returning")
                if "_tmp" in res.a.dims:
                    logger.error(f"_tmp {self.level_name}, {dims}")
                    exit()
                logger.info(f"STEP({self.level_name}, {dims}) Returning\nRes={res}")
                return res
                #1) Group arrays
                #2) Align it making a new data_array along the "_tmp" dimension
                #3) If the elements are also a dataarray, reapply  _compute_combine, but along _tmp
            else:
                if "_tmp" in res.a.dims:
                    logger.error("_tmp")
                    exit()
                return res
    
    def count_tmp(self, dims: Set[str]):
        return self._compute_combine(dims, lambda x, d: x.count(d) if len(d) > 0 else ~x.isnull(), lambda x, d: x.sum(d))
    

    def rename(self, n: str):
        r = self.copy()
        if isinstance(n, str):
            r.name = n
        else:
            r.name = n(self.name)
        return r
    def mean_tmp(self, dims: Set[str]):
        def leaf_func(x: xr.DataArray, d: Set[str]):
            if len(d)>0:
                count = x.count(d)
                mean = x.mean(d)
            else:
                count = ~x.isnull()
                mean = x

            res = xr.Dataset({"_count": count, "_mean":mean})
            res = res.to_array(dim="_aggdata")
            return res
        
        def gather_func(x: xr.DataArray, d: Set[str]):
            # in_df = x.to_dataframe().unstack("_aggdata").droplevel(0, axis=1)
            # print("INPUT=")
            # print(in_df)
            count = x.sel(_aggdata="_count", drop=True)
            prev_mean = x.sel(_aggdata="_mean", drop=True)
            sum = count.sum(d)
            mean = (prev_mean * count).sum(d)/sum
            res = xr.Dataset({"_count": sum, "_mean":mean})
            res = res.to_array(dim="_aggdata", name="toto")
            # out_df = res.to_dataframe().unstack("_aggdata").droplevel(0, axis=1)
            # expected_count = in_df["_count"].groupby("freq").sum()
            # expected_dot = in_df.groupby("freq").apply(lambda d: (d["_count"] * d["_mean"]).sum())
            # expected_mean = expected_dot/expected_count
            # out_df["expected_mean"] = expected_mean
            # out_df["expected_count"] = expected_count
            # print(f"OUTDF=\n{out_df}")
            # print(expected_count)
            # print(expected_mean)
            # input()
            return res
        
        return self._compute_combine(dims, 
            leaf_func, 
            gather_func,
        ).sel(_aggdata="_mean", drop=True).rename(lambda n: f"{n}.mean({dims})")
    
    def median_tmp(self, dims: Set[str]):
        def leaf_func(x: xr.DataArray, d: Set[str]):
            if len(d)>0:
                count = x.count(d)
                mean = x.median(d)
            else:
                count = ~x.isnull()
                mean = x

            res = xr.Dataset({"_count": count, "_median":mean})
            res = res.to_array(dim="_aggdata")
            return res
        
        def gather_func(x: xr.DataArray, d: Set[str]):
            # in_df = x.to_dataframe().unstack("_aggdata").droplevel(0, axis=1)
            # print("INPUT=")
            # print(in_df)
            count = x.sel(_aggdata="_count", drop=True)
            prev_median = x.sel(_aggdata="_median", drop=True)
            sum = count.sum(d)
            median = (prev_median * count).sum(d)/sum
            res = xr.Dataset({"_count": sum, "_median":median})
            res = res.to_array(dim="_aggdata", name="toto")
            # out_df = res.to_dataframe().unstack("_aggdata").droplevel(0, axis=1)
            # expected_count = in_df["_count"].groupby("freq").sum()
            # expected_dot = in_df.groupby("freq").apply(lambda d: (d["_count"] * d["_mean"]).sum())
            # expected_mean = expected_dot/expected_count
            # out_df["expected_mean"] = expected_mean
            # out_df["expected_count"] = expected_count
            # print(f"OUTDF=\n{out_df}")
            # print(expected_count)
            # print(expected_mean)
            # input()
            return res
        
        return self._compute_combine(dims, 
            leaf_func, 
            gather_func,
        ).sel(_aggdata="_mean", drop=True).rename(lambda n: f"{n}.mean({dims})")

    def _count(self, dims: Set[str], *, keep_attrs=None, **kwargs):
        curr_dims = {x for x in dims if x in self.a.dims}
        other_dims = {x for x in dims if not x in self.a.dims}
        i=0
        n=self.a.size
        def compute_embedded_count(x):
            nonlocal i,n
            print(f"count cur={curr_dims}, other={other_dims}, i={i}/{n}")
            i=i+1
            if isinstance(x, DataArray):
                return x._count(other_dims, keep_attrs=keep_attrs, **kwargs)
            else:
                return 0
        res = self.copy()
        if len(other_dims)>0:
            res.a = xr.apply_ufunc(compute_embedded_count, self.a, vectorize=True)
        
        # def combine_results(dl):
        #     notdata = {i:v for i,v in enumerate(dl.flat) if not isinstance(v, DataArray)}
        #     data = {i:v.a for i,v in enumerate(dl.flat) if isinstance(v, DataArray)}
        #     if len(data) > 0:
        #         try:
        #             r = xr.Dataset(data).to_array(dim="_tmp_count")
        #         except:
        #             logger.error(f"Error with data {data}")
        #             return 0
        #         return combine_results(dl)

        if len(curr_dims)>0:
            if len(self.subdims) ==0:
                res.a = res.a.count(curr_dims)
            else:
                # pass
                def agg_func(d):
                    # input(d)
                    notdata = {i:v for i,v in enumerate(d.flat) if not isinstance(v, DataArray)}
                    data = {i:v.a for i,v in enumerate(d.flat) if isinstance(v, DataArray)}
                    if len(notdata)>0:
                        input(f"NOTDATA\n{notdata}")
                    if len(data) > 0:
                        try:
                            r = xr.Dataset(data).to_array(dim="_tmp_count")
                        except:
                            print(f"Error with data {data}")
                            return 0
                        r = r.sum("_tmp_count")
                        return DataArray.from_xarray(r)
                    else:
                        return 0

                res.a = xr.apply_ufunc(agg_func, res.a, input_core_dims=[res.a.dims])
                res.subdims = set([])
        return res
    
    def count(self, dim=None, *, keep_attrs=None, **kwargs):
        if dim == None:
            dim = [...]
        elif isinstance(dim, str):
            dim = [dim]
        if ... in dim:
            dim = {x for x in set(self.a.dims).union(self.subdims) if not x in dim}
        else:
            dim = set(dim)
        return self._count(dim, keep_attrs=keep_attrs, **kwargs)
        
#Takes a dataarray with a dimension "tmp" and returns a new DataArray without the tmp column but a tmp column in each of its arrays
def tmp_func(d: DataArray, other_dims: Set[str]) -> DataArray:
    def f(dn: np.ndarray):
        dn = dn.reshape(list(dn.shape[:-1]) +[-1])
        def agg(d):
            data = {i:v.a for i,v in enumerate(d) if isinstance(v, DataArray)}
            if len(data) > 0:
                curr_dims = {x for x in data[0].dims if x in other_dims}
                res = d[0].copy()
                res.a = xr.apply_ufunc(lambda *x: np.stack(x, axis=-1), *data.values(), input_core_dims=[curr_dims for _ in range(len(data))], exclude_dims=[curr_dims for _ in range(len(data))],output_core_dims=[["tmp"]])
                #Here is where the recursive call should be
                return res
            else: 
                return np.nan
        return np.apply_along_axis(agg, len(dn.shape)-1, dn)
    xr.apply_ufunc(f, d.a, input_core_dims=[["tmp"]])

# output_dims: List[List[str]]=[[]]

import nptyping as npt

def rec_transform_array_for_ufunc(array: DataArray, agg_col, aggregated_dims: List[str], lowered_dims: List[str]) -> DataArray:
    if array.level_name is None:
        return array
    res = array.copy()
    lowered_dims = [x for x in array.a.dims if x in lowered_dims]
    #Create agg column in current array
    #Create agg column in each subarray
    #In each subarray group, stack the aggregated columns and merge the others and apply recursively
    def stack(*arrays, n):
        if n!=0:
            arrays = [ar.reshape(list(ar.shape[:-(n)]) +[-1]) for ar in arrays]
        else:
            arrays = [np.expand_dims(ar, -1) for ar in arrays]
        r = np.concatenate(arrays, axis=-1)
        return r
    def work(group: np.ndarray):
        group = [v for v in group if isinstance(v, DataArray)]
        if len(group) >0:
            representative = group[0]
            group = [v.a for v in group]
            curr_dims = {x for x in aggregated_dims if x in representative.a.dims}
            # if len(curr_dims) ==0:
            #     print(len(curr_dims))
            # if len(curr_dims) > 0:
            try:
                stacked_merged = xr.apply_ufunc(stack, *group, input_core_dims=[curr_dims for _ in group], exclude_dims=curr_dims, output_core_dims=[tuple([agg_col])], kwargs={"n": len(curr_dims)})
            except:
                print(len(curr_dims))
                raise
            # else:
            #     stacked_merged = xr.apply_ufunc(lambda *x: np.concatenate(x), *group)
            res = representative.copy()
            res.a = stacked_merged
            return rec_transform_array_for_ufunc(res, agg_col, aggregated_dims, lowered_dims)
        else:
            return np.nan
    def f(grouparr: np.ndarray):
        # dn = dn.reshape(list(dn.shape[:-(n-1)]) +[-1]) 
        res = np.apply_along_axis(work, -1, grouparr)
        # input("Waiting f")
        return res
    
    
    res.a = xr.apply_ufunc(f, array.a, input_core_dims=[lowered_dims+[agg_col]])
    return res

def transform_array_for_ufunc(array: DataArray, aggregated_dims: List[str], lowered_dims: List[str], agg_col ="__agg_transform_ufunc") -> DataArray:
    
    res=array.copy()
    curr_dims = {x for x in aggregated_dims if x in array.a.dims}
    def agg_column(x, n):
        return x.reshape(list(x.shape[:-n]) +[-1])
    if len(curr_dims) >0:
        res.a = xr.apply_ufunc(agg_column, res.a, input_core_dims=[curr_dims], kwargs={"n": len(curr_dims)}, output_core_dims=[[agg_col]])
    else:
        res.a = res.a.expand_dims(agg_col)
    # input("Waiting")
    return rec_transform_array_for_ufunc(res, agg_col, aggregated_dims, lowered_dims)
    
def rec_apply_ufunc(func, arrays, output_dims: List[List[str]] = [[]], agg_col ="__agg_transform_ufunc", lowered_dims=[]):
    arrays = [ar for ar in arrays if isinstance(ar, DataArray)]
    if len(arrays) ==0:
        return np.nan
    # else:
        # logger.info(len(arrays))
    res = arrays[0].copy()
    
    if res.level_name is None:
        def compute(*dn):
            logger.warning([d.shape for d in dn])
            # logger.info(f"len {len(dn)}")
            stacked = np.stack(dn, axis =-1)
            # input(stacked.shape)
            # reshaped = stacked.reshape(list(stacked.shape[:-1]) +[-1])
            # logger.warning(reshaped.shape)
            # input(reshaped.shape)
            # return np.apply_along_naxis(lambda x: func(*x), -1, stacked)
            return func(*dn)
        res.a = xr.apply_ufunc(compute, *[ar.a for ar in arrays], input_core_dims=[lowered_dims+[agg_col] for _ in arrays], exclude_dims={agg_col}, output_core_dims=output_dims)
        logger.warning(res.a)
    else:
        def rec_call(*x):
            # print([d.flat[0] for d in x])
            # input()
            return rec_apply_ufunc(func,  x, output_dims, agg_col=agg_col, lowered_dims=lowered_dims)
        res.a = xr.apply_ufunc(rec_call, *[ar.a for ar in arrays if isinstance(ar, DataArray)], vectorize=True)
    print(res)
    # input()
    return res

def apply_ufunc(func, *arrays, aggregated_dims: List[List[str]], lowered_dims: List[str], output_dims: List[List[str]] = [[]]):
    agg_col ="__agg_transform_ufunc"
    arrays = [transform_array_for_ufunc(a, agg, lowered_dims, agg_col=agg_col) for a,agg in zip(arrays, aggregated_dims)]
    return rec_apply_ufunc(func, arrays, output_dims, agg_col, lowered_dims=lowered_dims)
    
    # other_dims = {x for x in removed_dims if not x in array.a.dims}

    # #Grouping current dimensions
    # def grp_f(dn: np.ndarray):
    #     dn.
    # xr.apply_ufunc(grp_f, array.a, input_core_dims=[list(curr_dims.keys())], output_core_dims=[f"_agg_{set(curr_dims.values())}"])
    
    # #Passing dimensions to level (if necessary)

    # #Recursion