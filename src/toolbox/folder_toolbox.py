import pathlib
import pandas as pd
import logging
import numpy as np
from typing import List, Union, Callable, Any
import matplotlib.pyplot as plt
import matplotlib as mpl

logger=logging.getLogger(__name__)


def read_folder_as_database(
        search_folder: pathlib.Path,
        columns: List[str],
        pattern: Union[str, List[str]],
) -> pd.DataFrame:
    logger.info("Read folder as database called")
    if not isinstance(pattern, List):
        pattern = [pattern]
    if not search_folder.exists():
       logger.warning("search folder does not exist")
       return pd.DataFrame([],columns=["filename", "ext"]+ columns +["path"])
    raw_files=[]
    for p in pattern:
        raw_files += search_folder.glob(p)
    
    l=[]
    for file in raw_files:
        if file.parents[len(columns)]==search_folder:
            cols = [file.parents[i].name for i in range(len(columns))]
            cols.reverse()
            l.append([file.stem, file.suffix]+ cols + [str(file)])
        else:
            logger.debug("Ignored file {} because {} != {}".format(file, file.parents[len(columns)], search_folder))
    database = pd.DataFrame(l,columns=["filename", "ext"]+ columns +["path"])
    logger.info("Read folder as database done. {} elements".format(len(database.index)))
    return database

import re, tqdm

recup_wild = re.compile('{.+?}')
def extract_wildcards(strlist, pattern: str, file_col = "Files", tqdm = tqdm.tqdm):
    cols = []
    def msub(match):
        val = match.group()
        cols.append(val[1:-1])
        return '(?P<{}>.*)'.format(val[1:-1])
    regex_str = recup_wild.sub(msub, pattern)
    regex = re.compile(regex_str)
    res=[]
    for s in tqdm(strlist):
        d = regex.match(s)
        if not d is None:
          d=d.groupdict()
        else:
          d = {c:None for  c in cols}
          # d["path"] = s
          # print("dict: ", d)
        d[file_col] = s
        res.append(d)
    df = pd.DataFrame(res)
    for col in df.columns:
       df[col]=df[col].astype(str)
    return df

def files_as_database(config):
    base_folder = pathlib.Path(config["base_folder"])
    inputs_files_database_cache = config["inputs_files_database_cache"]
    results={}

    for name, f in inputs_files_database_cache.items():
        path = base_folder / pathlib.Path(f["path"])
        if not path.exists():
          if not f["recompute"]:
            logger.info("Database cache {} was marked as no recompute but inexistant. Recomputing".format(name))
            f["recompute"]=True
        if not f["recompute"]:
            results[name] = pd.read_csv(str(path), sep="\t")

    inputs=config["inputs"]
    for inputname, input in inputs.items():
        if (not input["out_dataframe"] in inputs_files_database_cache) or inputs_files_database_cache[input["out_dataframe"]]["recompute"]:
          input_files_folder = pathlib.Path(input["input_files_folder"])
          files=[str(f.relative_to(base_folder / input_files_folder)) for f in (base_folder / input_files_folder).glob("**/*")]
          def get_full_path(rel_path):
              return base_folder / input_files_folder / rel_path
          patterns = input["file_patterns"]
          for pattern in patterns:
              pattern_df = extract_wildcards(files, pattern)
              pattern_df["full_path"]=pattern_df.apply(
                  lambda row: str(get_full_path(pathlib.Path(row["rel_path"]))),
                  axis=1
              )
              transformed_df = pd.DataFrame()
              def make_col(str, wildcards):
                  return str.format(**wildcards)
              for colname, colval in input["columns"].items():
                  transformed_df[colname] = pattern_df.apply(lambda row: make_col(str(colval), row.to_dict()), axis=1)
          if not input["out_dataframe"] in results:
              results[input["out_dataframe"]] = pd.DataFrame()
          results[input["out_dataframe"]] = pd.concat([results[input["out_dataframe"]], transformed_df], ignore_index=True)

    for name, f in inputs_files_database_cache.items():
        if f["recompute"]:
          path = base_folder / pathlib.Path(f["path"])
          path.parent.mkdir(parents=True, exist_ok=True)  
          results[name].to_csv(str(path), sep="\t", index=False)

    return results

def database_select(db: pd.DataFrame, selector):
  """Selector is a dictionary in the form (null entries may be omitted)

    filter: null | str  #string passed to pandas.query
    sort_by: null | 
      xxx: "asc" | "desc"
      yyy: "asc" | "desc"
    partition_by: null | ["subject", "structure"] 
    select_range_per_partition: null | [start, end] #end is not included
  """
  # logger.info("called")
  res = db.copy()
  if "filter" in selector and selector["filter"]:
    res.query(selector["filter"], inplace=True)

  if "sort_by" in selector and selector["sort_by"]:
    cols = [col for col in selector["sort_by"].keys()]
    order = [True if order=="asc" else False for order in selector["sort_by"].values()]
    res.sort_values(by=cols, ascending=order, inplace=True)  

  if "__group_rank" in res.columns:
    logger.error("__group_rank is a reserved column for database_select. It will be rewritten.")

  if "partition_by" in selector and selector["partition_by"]:
    res["__group_rank"]=res.groupby(selector["partition_by"]).cumcount()
  else:
    res["__group_rank"]=0
  
  if "select_range_per_partition" in selector and selector["select_range_per_partition"]:
    start = selector["select_range_per_partition"][0]
    end = selector["select_range_per_partition"][1]
    res=res[res["__group_rank"].between(start, end, inclusive="left")]
  
  res.drop(columns=["__group_rank"], inplace=True)
  return res


    
# Based on https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
import unicodedata
import string

valid_filename_chars = "-_()= %s%s" % (string.ascii_letters, string.digits)
char_limit = 255

def clean_filename(filename, whitelist=valid_filename_chars, replace=' '):
    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')
    
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]    

import tqdm, os
def find_files_recursively(path, tqdm=None):
    files = []
    # total=1 assumes `path` is a file
    if not tqdm is None:
      t = tqdm(desc="Finding recursive files", total=1, unit="file", disable=False)
    else:
       t = tqdm.tqdm(total=1, unit="file", disable=True)
    if not os.path.exists(path):
        raise IOError("Cannot find:" + path)

    def append_found_file(f):
        files.append(f)
        t.update()

    def list_found_dir(path):
        """returns os.listdir(path) assuming os.path.isdir(path)"""
        listing = os.listdir(path)
        # subtract 1 since a "file" we found was actually this directory
        t.total += len(listing) - 1
        # fancy way to give info without forcing a refresh
        t.set_postfix(dir=path[-10:], refresh=False)
        t.update(0)  # may trigger a refresh
        return listing

    def recursively_search(path):
        if os.path.isdir(path):
            for f in list_found_dir(path):
                recursively_search(os.path.join(path, f))
        else:
            append_found_file(path)

    recursively_search(path)
    t.set_postfix(dir=path)
    t.close()
    return files



import ast
class DataPath:
    file_path: str
    keys: List[Any]

    def __init__(self, file_path, keys=[]):
        self.file_path = file_path
        self.keys = keys

    def __str__(self):
        if len(self.keys) > 0:
            return "File({})[{}]".format(self.file_path, ", ".join([str(key) for key in self.keys]))
        else:
            return "File({})".format(self.file_path)
        
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def from_str(s):
       r = re.compile("File\(([^)]+)\)(.*)")
       d = r.fullmatch(s).groups()
       file = d[0]
       if len(d[1]) ==0:
          keys=[]
       else:
          keys=d[1][1:-1].split(", ")
       return DataPath(file, keys)
    
