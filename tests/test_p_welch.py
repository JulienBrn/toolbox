import pathlib
import pandas as pd
import logging
import beautifullogger
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
# import pickle
import matplotlib
import joblib
import toolbox
# import seaborn as sns
import scipy
import tqdm

logger=logging.getLogger(__name__)
beautifullogger.setup()
matplotlib.use("tkagg")

res_folder = pathlib.Path(sys.argv[0]).parent / "Results"
analysis_files_folder = pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MonkeyData4Review/")
cache_folder= pathlib.Path(sys.argv[0]).parent / ".cache"

memory = joblib.Memory(cache_folder, verbose=0)


file_db = memory.cache(toolbox.read_folder_as_database)(
    search_folder=analysis_files_folder, 
    pattern = "**/*.mat",
    columns=["Condition", "Subject", "Structure", "Date"])

plots_db=file_db.merge(pd.DataFrame([["lfp"], ["mu"]], columns=["signal"]), how="cross")
plots_db=plots_db.merge(pd.DataFrame([[True], [False]], columns=["normalized"]), how="cross")
plots_db=plots_db.merge(pd.DataFrame([[True], [False]], columns=["rm_artefacts"]), how="cross")


def get_data_col_key(axis, df, cols=["filename", "signal", "normalized", "rm_artefacts"]):
    def as_str(x):
        if isinstance(df, pd.DataFrame):
            return x.astype(str)
        else:
            return str(x)
        
    res=axis

    for col in cols:
        res+="_"+ as_str(df[col])
    
    return res
    
plots_db["x_data_col"] = get_data_col_key("x", plots_db)
plots_db["y_data_col"] = get_data_col_key("y", plots_db)

plots_db = plots_db[(plots_db["rm_artefacts"]==True) & (plots_db["normalized"]==True)]
print(plots_db)

def compute_result(path, fs, signal, window, normalize=True, rm_artefacts=True):
    def get_sig(path, fs, signal):
        raw = scipy.io.loadmat(path)["RAW"][0,]
        if signal == "lfp":
            sig, nfs = toolbox.extract_lfp(raw, fs)
        elif signal == "mu":
            sig, nfs = toolbox.extract_mu(raw, fs)
        return sig,nfs
    sig,nfs = memory.cache(get_sig)(path, fs, signal)

    if normalize:
        sig = (sig - sig.mean())/sig.std()

    ret_x, ret_y = memory.cache(scipy.signal.welch)(sig, nfs, nperseg=window*nfs)

    if rm_artefacts:
        memory.cache(toolbox.remove_artefacts)(ret_y)

    return ret_x, ret_y


result_computer = memory.cache(compute_result)
data = {
    get_data_col_key(axis, row):arr
    for i, (_,row) in zip(tqdm.trange(len(plots_db)), (plots_db.iterrows()))
    for axis, arr in zip(["x", "y"], result_computer(row["path"], 25000, row["signal"], 3, normalize=row["normalized"], rm_artefacts=row["rm_artefacts"])) 
}

toolbox.add_draw_metadata(plots_db, 
                        fig_group=["normalized", "rm_artefacts"],
                        row_group=["Condition", ],
                        col_group=["Structure", "signal"],
                        # color_group=["Condition"],
)

plotcanvas = toolbox.prepare_figures(plots_db)

plotcanvas.plot(plots_db, data)
plt.show()