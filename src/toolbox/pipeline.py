from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, List, Literal, IO, Callable
import pathlib
import logging
import numpy as np
import sys
from toolbox.ressource_manager import Manager, RessourceLoader, RessourceHandle

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
  if hasattr(df, "progress_apply"):
    return df.progress_apply(compute_elem, axis=1)
  else:
    return df.apply(compute_elem, axis=1)

def get_columns(df, columns):
  if isinstance(columns, str):
      columns=[columns]
  def compute_and_clean_row(row):
      dres={}
      for col in columns:
        if isinstance(row[col], RessourceHandle):
          res = row[col].get_result()
        else:
          res = row[col]
        if hasattr(res, "shape"):
          dres[col] = "Shape{}".format(res.shape)
        elif hasattr(res, "__len__"):
          dres[col] = "{}_of_{}_elements".format(type(res), len(res))
        else:
            dres[col] = res
        if isinstance(row[col], RessourceHandle):
          row[col].save()
      for col in row.index:
          if isinstance(row[col], RessourceHandle):
            row[col].unload()
      return dres
  df[columns] = df.progress_apply(compute_and_clean_row, axis=1, result_type="expand")
  result_df = df[[col for col in df.columns if not isinstance(df[col].iat[0], RessourceHandle)]]
  return result_df