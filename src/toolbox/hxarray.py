from __future__ import annotations
from typing import List, Dict, Any, Set
import xarray as xr, numpy as np, logging

logger = logging.getLogger(__name__)

class DataArray:
    a: xr.DataArray
    level_name: str
    subdims: Set[str]
    subcoords: Set[str]

    def __init__(self):pass
    def copy(self):
        r = DataArray()
        r.a = self.a.copy()
        r.level_name = self.level_name
        r.subdims = self.subdims.copy()
        r.subcoords = self.subcoords.copy()
        return r

    @staticmethod
    def from_xarray(a: xr.DataArray) -> DataArray:
        r = DataArray()
        r.a = a
        r.subdims = {}
        r.level_name = None
        r.subcoords = {}
        return r
    
    def from_xarray_level(a: xr.DataArray, level_name: str) -> DataArray:
        xr.DataArray().sel()
        r = DataArray()
        r.a = a
        r.level_name = level_name
        if a.size == 0:
            r.subdims =set({})
            r.subcoords =set({})
        else:
            # print(a)
            subdims = xr.apply_ufunc(lambda x: set(x.a.dims).union(x.subdims) if isinstance(x, DataArray) else {None}, a, vectorize=True).values.flat
            subcoords: np.ndarray = xr.apply_ufunc(lambda x: set(x.a.coords).union(x.subcoords) if isinstance(x, DataArray) else {None}, a, vectorize=True).values.flat
            # print(subcoords)
            s = subdims[0]
            # print(s)
            if ((subdims == s) | (subdims=={None})).all():
                r.subdims = s
            else:
                raise ValueError(f"Different subdims. Examples:\n{subdims[subdims != s]}")
            
            r.subcoords = set.intersection(*[x for x in subcoords if not x =={None}])
        return r
    
    def __str__(self):
        s = f"""
Dimensions: {', '.join([f'{d}({s})' for d,s in self.a.sizes.items()])}, {', '.join(self.subdims)}
{self.a.coords}
Embedded coordinates: {', '.join(self.subcoords)}
Values {self.a}""".strip()
        return s
    
    def sel(self, indexers: Dict[str, Any]=None, method=None, tolerance=None, drop=False, **indexers_kwargs):
        if indexers is None:
            indexers = indexers_kwargs
        cur_indexers = {k:v for k,v in indexers.items() if k in self.a.dims}
        other_indexers = {k:v for k,v in indexers.items() if not k in self.a.dims}
        res = self.copy()
        print(self.a)
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
            res.a = xr.apply_ufunc(lambda x: compute_embedded_sel, self.a.sel(cur_indexers, method, tolerance, drop), vectorize=True)
        return res
    
    def _push_dim_to_one_level_down(self, dim):
        if len(self.subdims) ==0:
            return self
        def agg_func(dn):
            logger.info(f"entered {dn.shape}")
            def agg(d):
                logger.info(f"agg {d.shape}")
                data = {i:v.a for i,v in enumerate(d) if isinstance(v, DataArray)}
                if len(data) > 0:
                    r = xr.Dataset(data).to_array(dim=dim)
                    return DataArray.from_xarray_level(r, self.level_name)
                else:
                    return np.nan
            return np.apply_along_axis(agg, len(dn.shape)-1, dn)
        res = self.copy()
        res.a = xr.apply_ufunc(agg_func, self.a, input_core_dims=[[dim]])
        res.subdims.add(dim)
        return res
    
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
        

