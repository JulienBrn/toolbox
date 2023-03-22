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
    return database

import re

recup_wild = re.compile('{.+?}')
def extract_wildcards(strlist, pattern):
    print("pattern: ", pattern)
    def msub(match):
        val = match.group()
        print(val)
        return '(?P<{}>.*)'.format(val[1:-1])
    regex_str = recup_wild.sub(msub, pattern)
    regex = re.compile(regex_str)
    res=[]
    for s in strlist:
        d = regex.match(s)
        if not d is None:
          d=d.groupdict()
          d["rel_path"] = s
          # print("dict: ", d)
          res.append(d)
    df = pd.DataFrame(
        res, 
        columns=["rel_path", "condition", "subject", "structure", "date", "unit"]
      )
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

class FileData:
    load_func: Callable[[str], Any]

    def __init__(path, select = None, reshape = None, nentries=None):
        pass

def load(path: pathlib.Path):
    pass #todo

def load_paths(f):
    def new_f(*args, **kwargs):
        nargs = [arg if not isinstance(arg, FileData) else load(arg) for arg in args]
        nkwargs = {key:arg if not isinstance(arg, FileData) else load(arg) for key,arg in kwargs}
        return f(*nargs, **kwargs)
    return new_f
