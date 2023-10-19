import pandas as pd, numpy as np, functools, xarray as xr, dask, dask.dataframe as dd
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from dask.diagnostics import ProgressBar




logger = logging.getLogger(__name__)
beautifullogger.setup(logmode="w")
logging.getLogger("fsspec.local").setLevel(logging.WARNING)
for x in ["utils_comm", "core", "worker", "scheduler", "nanny"]:
    logging.getLogger(f"distributed.{x}").setLevel(logging.WARNING)
logging.getLogger("distributed.core").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
ProgressBar().register()

def aggregate_by(self, by, *args, **kwargs):
    all_index_cols={"Species", "Structure", "Healthy", "signal_resampled_type", "t", "freq", "sig_num"}
    return self.groupby([col for col in self.columns if not col in by and col in all_index_cols], *args, **kwargs, observed=True)
dd.DataFrame.aggregate_by = aggregate_by

steps_t = Literal["from_pickle_database", "from_parquet_database", "from_merged_database"]
steps = ["from_pickle_database", "from_parquet_database", "from_merged_database"]
start: steps_t = "from_parquet_database"
nwindows = 5
nsignals = 2

if __name__ == "__main__":
    # from dask.distributed import Client
    # client = Client()
    if steps.index(start) <= steps.index("from_pickle_database"):
        logger.info("Creating parquet database")
        data_path = pathlib.Path(f"/home/julien/Documents/spectrogram_export/database.tsv")
        df: pd.DataFrame = toolbox.df_loader.load(data_path)
        df["sig_type"] = df.pop("signal_resampled_type")
        df["fullpath"] = df["path"] + df["suffix"]

        (n, k)=0,0
        def reexport_data(path:pathlib.Path):
            global n,k
            path = path.iat[0]
            path = pathlib.Path("/home/julien/Documents/" ) / pathlib.Path(path).relative_to("/home/julien/Documents/GUIAnalysis/")
            d: List[pd.DataFrame] = toolbox.pickle_loader.load(path)
            def mk_new_df(df, i):
                global n
                df.index.name = "t"
                df.columns.name=("freq")
                df: pd.DataFrame = pd.DataFrame({"amp" : df.stack()})
                newpath = pathlib.Path("." ) / "DataParquet"/ path.relative_to(pathlib.Path("/home/julien/Documents/" )).parent / (path.stem + f"_sig{i}" '.parquet' )
                if newpath.exists():
                    if newpath.is_file():
                        newpath.unlink(missing_ok=True)
                    else:
                        import shutil
                        shutil.rmtree(str(newpath))
                        # newpath.rmdir()
                newpath.parent.mkdir(exist_ok=True, parents=True)
                df["new_path"] = str(newpath)
                if n==0:
                    logger.info(f"Example of returned subdataframe\nIndexes are {df.index.names}\nColumns are {df.columns}. Dataframe is \n{df}")
                    logger.info(f"Exporting to {newpath}")
                
                df = df.reset_index()
                df.columns = ['t', 'freq', 'amp', 'new_path']
                # df=pd.DataFrame(df.values, columns=df.columns)
                # df=df.set_index(["freq", "t"])
                # input(df)
                # df.copy().to_parquet(newpath, engine="fastparquet")
                # ddf = dd.from_pandas(df, npartitions=1)
                # pickle.dump(ddf, open(newpath, "wb"))
                df.to_parquet(newpath, index=False)
                if n==0:
                    ddask = dd.read_parquet(newpath, columns=['t', 'freq', 'amp', 'new_path'])
                    # ddask =pickle.load(open(newpath, "rb"))
                    logger.info(f"Example of returned subdataframe after dask load\nColumns are {ddask.columns}. Dataframe is \n{ddask.head(5)}")
                n+=1
                return str(newpath)
            d= [mk_new_df(df, i) for i,df in enumerate(d)]
            r = pd.DataFrame([[i, x] for i,x in enumerate(d)], columns=["sig_num", "new_path"])
            r = r.set_index("sig_num")
            if k==0:
                logger.info(f"Example of returned groupby series\nIndexes are {r.index.names}\nColumns are {r.columns}. Dataframe is \n{r}")
            k+=1
            return r

        tqdm.tqdm.pandas(desc="Loading data files")
        init_cols = list(set(df.columns) -  {"path", "suffix", "column", "Ressource", "fullpath"})
        for col in init_cols:
            if df[col].dtype == object:
                df[col] = df[col].astype("category")
        df.set_index(init_cols, inplace=True)
        logger.info(f"Main pickle database df\nIndexes are {df.index.names}\nColumns are {df.columns}. Dataframe is \n{df}")
        res = df.groupby(init_cols, observed=True)["fullpath"].progress_apply(reexport_data)
        logger.info(f"Main parquet database df\nIndexes are {res.index.names}\nColumns are {res.columns}. Dataframe is \n{res}")
        res=res.reset_index()
        logger.info(f"Main parquet database df after reset_index\nIndexes are {res.index.names}\nColumns are {res.columns}. Dataframe is \n{res}")
        res.to_parquet("./DataParquet/all.parquet")

    if steps.index(start) <= steps.index("from_parquet_database"):
        logger.info("Creating merged database")
        import dask.dataframe as dd
        files_df: dd.DataFrame = dd.read_parquet("./DataParquet/all.parquet")
        if nsignals is not None:
            files_df = files_df.aggregate_by("sig_num").apply(lambda x: x.sample(nsignals) if len(x.index) > nsignals else x)
        logger.info(f"Main parquet df read\nColumns are {files_df.columns}. Dataframe is \n{files_df.head(5)}. NPartitions={files_df.npartitions}")

        all_df = files_df
        other_dfs=[]
        import tqdm
        prog = tqdm.tqdm(total=4214, desc="declaring parquet files")
        def load(x):
            global prog
            x=str(pathlib.Path(str(x)).with_suffix(".parquet"))
            if pathlib.Path(str(x)).exists():
                prog.update(1)
                d = dd.read_parquet(x, columns=['t', 'freq', 'amp', 'new_path'])
                if nwindows is not None:
                    def get_times(d):
                        if len(d.index) < nwindows:
                            return d
                        else:
                            x = np.random.randint(0, len(d.index)-nwindows)
                            return d.iloc[x:x+nwindows, :]
                    d = d.groupby("freq").apply(get_times)
                other_dfs.append(d) 
                
                # other_dfs.append(pickle.load(open(x, "rb"))) 
                # input(f"File {x} has columns {other_dfs[-1].columns}")
            else:
                logger.warning(f"Not found {x}")
            return 2
        files_df["new_path"].apply(load).compute()
        logger.info(f"Loaded other_dfs, got list of length {len(other_dfs)}\nExample df\nColumns are {other_dfs[0].columns}. Dataframe is \n{other_dfs[0].head(5)}")
        r = dd.concat(other_dfs)
        logger.info(f"Concatenated other_dfs.\nColumns are {r.columns}. Dataframe is \n{r.head(5)}")
        # r=r.reset_index()
        # logger.info(f"Reset index.\nColumns are {r.columns}. Dataframe is \n{r.head(5)}")
        # input()
        # for df in other_dfs:
        #     input(all_df)
        #     all_df = all_df.merge(df, on="new_path")
        merged = all_df.merge(r, on="new_path")
        logger.info(f"Merged\nColumns are {merged.columns}. Dataframe is \n{merged.head(5)}")
        # input()
        logger.info(f"Exporting merged files to ./Dataset/database_spectrogram_{nwindows}, {nsignals}.parquet")
        merged.to_parquet(f"./Dataset/database_spectrogram_{nwindows}, {nsignals}.parquet")

    # if steps.index(start) <= steps.index("from_merged_database"):
    #     logger.info("Creating reduced database")
    #     merged = dd.read_parquet("./Dataset/database_spectrogram_all.parquet")
    #     # merged = merged.head(1000000, compute=False, npartitions=10)
    #     logger.info(f"Merged\nIndexes are {merged.index.name}\nColumns are {merged.columns}. Dataframe is \n{merged.head(5)}")
    #     small = merged.aggregate_by(["t"]).apply(lambda x: x.head(500)).reset_index(drop=True)
    #     logger.info(f"Smalltmp df\nIndexes are {small.index.name}\nColumns are {small.columns}. Dataframe is \n{small.head(5)}")
    #     small = small.aggregate_by(["sig_num"]).apply(lambda x: x.head(50)).reset_index(drop=True)
    #     logger.info(f"Small df\nIndexes are {small.index.name}\nColumns are {small.columns}. Dataframe is \n{small.head(5)}")
    #     small.to_parquet("./Dataset/database_spectrogram_small.parquet")