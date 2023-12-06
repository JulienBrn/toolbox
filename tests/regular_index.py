from collections.abc import Mapping
from typing import Any
import xarray as xr, math, numpy as np

from xarray.core.indexes import IndexVars
from xarray.core.variable import Variable

class RegularIndex(xr.Index):
    def __init__(self, start, end, fs, name="unknown"):
        # print(f"Created with start={start}, end={end}, fs={fs}")
        self.start = start
        self.n = np.round((end-start)*fs)
        self.fs = fs
        self.name=name
        # print(f"n is={self.n}, end={self.end}")

    def equals(self, other):
        return self.start == other.start and self.n == other.n and self.fs == other.fs
    
    def sel(self, labels):
        if len(labels) !=1:
            raise NotImplementedError("RegularIndex sel")
        dim = labels.keys()[0]
        val = float(labels.values()[0])
        n = np.round((val-self.start)*self.fs)
        return xr.IndexSelResult(dim_indexers={dim:n})
    
    def create_variables(self, variables: Mapping[Any, Variable] | None = None) -> IndexVars:
        return {"t":xr.IndexVariable("t", np.array([]))}
        # print("called")
        # return {"t": xr.Variable(dims="t", data = self.as_array())}

    def join(self, other, how="inner"):
        if other.fs != self.fs:
            raise NotImplementedError("Joins with different fs")
        if how=="inner":
            return RegularIndex(max(self.start, other.start), min(self.end, other.end), self.fs)
        elif how=="outer":
            return RegularIndex(min(self.start, other.start), max(self.end, other.end), self.fs)
        elif how=="left":
            return self
        elif how=="right":
            return other
        else:
            raise NotImplementedError(f"Not implemented join method {how}")

    @classmethod
    def from_variables(cls, variables, options=None):
        # print(variables)
        assert len(variables) == 1

        values = list(variables.values())[0].to_numpy()
        if ((np.diff(values) - np.diff(values)[0])<0.000001).all():
            return cls(values[0], values[-1], 1/np.diff(values)[0])
        else:
            raise Exception("Not regular...")

    

    @property
    def end(self):
        return self.start + self.n/self.fs
    
    def as_array(self):
        return self.start+ np.arange(self.n)/self.fs
    
    def _repr_inline_(self, max_width):
        return f"Regular({self.start}, {self.end}, {self.fs})"