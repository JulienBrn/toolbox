import pandas as pd, numpy as np
from typing import Dict
import logging

logger=logging.getLogger(__name__)

def group_and_combine_old(df: pd.DataFrame, group_cols, include_eq = False):
  df = df.copy()
  df["__group_num"] = df.groupby(by=group_cols).ngroup()
  groups = df.groupby(by=group_cols+["__group_num"])
  def cross_merge(df: pd.DataFrame):
    df["__num_in_grp"] = np.arange(df.shape[0])
    df.drop(columns=group_cols, inplace=True)
    r = pd.merge(df, df, how="cross", suffixes=["_1", "_2"])
    if include_eq:
       r = r.loc[r["__num_in_grp_1"] <=  r["__num_in_grp_2"], :]
    else:
      r = r.loc[r["__num_in_grp_1"] <  r["__num_in_grp_2"], :]
    return r
  if hasattr(groups, "progress_apply"):
    ret: pd.DataFrame = groups.progress_apply(cross_merge)
  else: 
    ret: pd.DataFrame = groups.apply(cross_merge)
  ret = ret.reset_index()
  ret.drop(columns=["level_{}".format(len(group_cols)+1)], inplace=True)
  return ret

def group_and_combine(df: pd.DataFrame, group_cols, include_eq = False):
  logger.info("called")
  with_index = df.copy()
  with_index["__my_index_for_group_combine"] = np.arange(0, len(df.index))
  ret = pd.merge(with_index, with_index, on=group_cols, suffixes=["_1", "_2"])
  logger.info("done")
  if include_eq:
    return ret[ret["__my_index_for_group_combine_1"] <= ret["__my_index_for_group_combine_2"]].drop(columns=["__my_index_for_group_combine_1", "__my_index_for_group_combine_2"])
  else:
    return ret[ret["__my_index_for_group_combine_1"] < ret["__my_index_for_group_combine_2"]].drop(columns=["__my_index_for_group_combine_1", "__my_index_for_group_combine_2"])
  # df = df.copy()
  # df["__group_num"] = df.groupby(by=group_cols).ngroup()
  # groups = df.groupby(by=group_cols+["__group_num"])
  # def cross_merge(df: pd.DataFrame):
  #   df["__num_in_grp"] = np.arange(df.shape[0])
  #   df.drop(columns=group_cols, inplace=True)
  #   r = pd.merge(df, df, how="cross", suffixes=["_1", "_2"])
  #   if include_eq:
  #      r = r.loc[r["__num_in_grp_1"] <=  r["__num_in_grp_2"], :]
  #   else:
  #     r = r.loc[r["__num_in_grp_1"] <  r["__num_in_grp_2"], :]
  #   return r
  # if hasattr(groups, "progress_apply"):
  #   ret: pd.DataFrame = groups.progress_apply(cross_merge)
  # else: 
  #   ret: pd.DataFrame = groups.apply(cross_merge)
  # ret = ret.reset_index()
  # ret.drop(columns=["level_{}".format(len(group_cols)+1)], inplace=True)
  # return ret


# def transpose_into_rows(df, new_col_value_name, new_col_select_name, d):
#   dfs=[]
#   for col, name in d.items():
#     dfcol = df.copy()
#     dfcol[new_col_select_name] = name
#     dfcol[new_col_value_name] = df[col]
#     dfcol.drop(columns=d.items())
#     dfs.append(dfcol)
#   return pd.concat(dfs, ignore_index=True)
# def dataframe_reshape(df: pd.DataFrame, new_type_name, value_cols: Dict[str, Dict[str, str]]):
#   dfs=[]
#   for ntype, value_cols in value_cols.items():
#     dfcol = df.copy()
#     dfcol[new_type_name] = ntype
#     for (value_col_name, data_col) in value_cols.items():
#       dfcol[value_col_name] = df[data_col]
#     dfcol.drop(columns=value_cols.values())
#     dfs.append(dfcol)
#   return pd.concat(dfs, ignore_index=True)

def dataframe_reshape(df: pd.DataFrame, new_type_name, reshape: pd.DataFrame):
  dfs=[]
  for ntype, row in reshape.iterrows():
    dfcol = df.copy()
    dfcol[new_type_name] = ntype
    for col_name, data_col in row.items():
      dfcol[col_name] = df[data_col]
    dfs.append(dfcol)
  ret = pd.concat(dfs, ignore_index=True)
  ret.drop(columns=reshape.values.flatten(), inplace=True)
  return ret