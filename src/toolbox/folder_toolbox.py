import pathlib
import pandas as pd
import logging
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
import matplotlib as mpl

logger=logging.getLogger()


def read_folder_as_database(
        search_folder: pathlib.Path,
        columns: List[str],
        pattern: Union[str, List[str]],
):
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

