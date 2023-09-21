import datetime
from typing import Any
import numpy as np

#Based on https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object
def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def unique_id(v: Any):
   if isinstance(v, list):
         return f'[{",".join([mhash(x) for x in v])}]'
   elif isinstance(v, dict):
         return f'dict({",".join([f"{mhash(k)}={mhash(val)}" for k,val in sorted(v.items())])})'
   elif isinstance(v, str):
         return v
   elif isinstance(v, int) or isinstance(v, float) or isinstance(v, np.int64):
         return f"{str(v)}: {type(v)}"
   else:
         raise Exception(f"Impossible to hash {v} of type {type(v)}")