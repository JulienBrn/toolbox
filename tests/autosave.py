import logging, pathlib, pickle, dask.dataframe as dd, pandas as pd

logger=logging.getLogger(__name__)


class Autosave:
    nwindows=500
    nsignals=50
    result_base_path = "./Results"
    def __init__(self, name, version, debug=False, result_base_path=None):
        if not result_base_path is None:
            self.result_base_path = result_base_path
        if not "param_name" in Autosave.__dict__:
            self.param_name = "all" if (self.nwindows,self.nsignals) == (None,None) else f"{self.nwindows}, {self.nsignals}" 

        self.result_path = pathlib.Path(f"{self.result_base_path}/{name}/Data/from_spectrogram_{self.param_name}, {version}.pickle")
        self.name=name
        self.debug = debug
    def __call__(self, f):
        def new_f(*args, **kwargs):
            if not self.result_path.exists():
                logger.info(f"Computing {self.name}")
                res = f(*args, **kwargs)
                if isinstance(res, dd.DataFrame) or isinstance(res, dd.Series):
                    logger.info(f"{self.name} defined as a dask element." + "" if not self.debug else " Computing extract...")
                    if self.debug:
                        logger.info(f"Extract is\n{res.head(5)}")
                    return res
                elif not self.debug:
                    logger.info(f"{self.name} computed, now saving")
                    self.result_path.parent.mkdir(exist_ok=True, parents=True)
                    pickle.dump(res, self.result_path.open("wb"))
                    logger.info(f"{self.name} saved")
            else:
                logger.info(f"Loading {self.name}")
                res= pickle.load(self.result_path.open("rb"))
            logger.info(f"{self.name} is\n{res}")
            if isinstance(res, pd.DataFrame):
                self.result_path.parent.mkdir(exist_ok=True, parents=True)
                res.to_csv(self.result_path.with_suffix(".tsv"), sep="\t")
            return res
        return new_f