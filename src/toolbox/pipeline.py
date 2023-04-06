from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, List, Literal, IO, Callable
import pathlib
import logging
import numpy as np
import sys
from toolbox.ressource_manager import Manager, RessourceLoader

def mk_block(
  df: pd.DataFrame, 
  params: List[str], #Columns to use as params 
  func, #The function to call
  out: Dict[Any, Tuple[RessourceLoader, str, bool | None]] | Tuple[RessourceLoader, str, bool | None], #The loader and name of the output columns
  manager: Manager
):
  def compute_elem(row):
    nonlocal params
    params = {key:val for key, val in row.items() if key in params}
    if not type(out) is tuple:
      ressource_dict = manager.declare_computable_ressources(func, params, out)
    else:
      ressource = manager.declare_computable_ressource(func, params, *out)
      ressource_dict = {out[1]: ressource}
    return pd.concat([row,pd.Series(ressource_dict)])
   
  return df.apply(compute_elem, axis=1)