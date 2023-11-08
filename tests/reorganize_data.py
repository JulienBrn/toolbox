import pandas as pd, numpy as np, functools, random
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger

logger = logging.getLogger(__name__)
beautifullogger.setup()

nwindows, nsignals = 500, 50
nwindows, nsignals = None, None
param_name = f"{nwindows if not nwindows is None else 'all'},{nsignals if not nsignals is None else 'all'}"


def make_data():
    logger.info("Creating parquet database")
    data_path = pathlib.Path(f"/home/julien/Documents/full_spectrogram/database.tsv")
    df: pd.DataFrame = toolbox.df_loader.load(data_path)
    df["sig_type"] = df.pop("signal_resampled_type")

    tqdm.tqdm.pandas(desc="Computing path of data files")
    df["path"] = df.apply(
        lambda row: pathlib.Path("/home/julien/Documents/" ) / pathlib.Path(row["path"].replace("full_pwelch", "full_spectrogram") + row["suffix"]).relative_to("/home/julien/Documents/GUIAnalysis/"),
        axis=1
    )

    tqdm.tqdm.pandas(desc="Renaming some metadata")
    df["Condition"] = df["Healthy"].progress_apply(lambda x: "Control" if x else "Park")
    df["CorticalState"] = df["Species"].progress_apply(lambda x: "Anesthetized" if x=="Rat" else "Awake" if x=="Human" else "Awake" if x=="Monkey" else "Unknown")


    df= df[["Species", "Structure", "Condition", "CorticalState", "sig_type", "path"]]

    for col in ["Species", "Structure", "Condition", "CorticalState", "sig_type"]:
        df[col] = df[col].astype("category")

    tqdm.tqdm.pandas(desc="Reexporting")
    df = df.groupby(["Species", "Structure", "Condition", "CorticalState", "sig_type"], observed=True).progress_apply(reexport_data)
    print(df)
    print(df["square_size"].sum()/10**6, df["list_size"].sum()/10**6, df["list_size"].sum()/df["square_size"].sum())
    df.to_parquet(pathlib.Path("." ) / "DataParquetCustom"/ f"{param_name}.parquet")

def reexport_data(row):
    row: pd.Series = row.iloc[0, :]
    d: List[pd.DataFrame] = toolbox.pickle_loader.load(row["path"])
    if not nsignals is None:
        if nsignals < len(d):
            d = random.sample(d, nsignals)
    if not nwindows is None:
        d = [df.iloc[0:nwindows, :] if len(df.index) >nwindows else df for df in d]
    freq_sig_df = pd.Series(d).apply(lambda df: pd.Series([df[freq] for freq in df.columns], index=df.columns))
    freq_sig_df.columns.name = "freq"
    def tmp(c):
        l = {i:s for i,s in c.to_dict().items()}
        r = pd.concat(l, axis=1)
        r.columns.name = "sig_num"
        r.index.name = "t"
        square_size = r.values.size
        list_size = int(r.count().sum())
        efficiency = list_size/square_size
        newpath: pathlib.Path = pathlib.Path("." ) / "DataParquetCustom"/ f"{param_name}" /row["path"].relative_to(pathlib.Path("/home/julien/Documents/" )).parent / (row["path"].stem + f"_freq{c.name}.parquet")
        # input(newpath)
        newpath.parent.mkdir(exist_ok=True, parents=True)
        r.to_parquet(newpath)
        return pd.Series(dict(newpath=str(newpath), square_size=square_size, list_size=list_size, efficiency=efficiency))
    freq_df = freq_sig_df.apply(tmp).T
    return freq_df

if __name__=="__main__":
   make_data()
