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
# import joblib
import toolbox
# import seaborn as sns

logger=logging.getLogger(__name__)
beautifullogger.setup()
matplotlib.use("tkagg")

res_folder = pathlib.Path(sys.argv[0]).parent / "Results"
analysis_files_folder = pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MonkeyData4Review/")
cache_folder= pathlib.Path(sys.argv[0]).parent / "cache"

# memory = joblib.Memory(cache_folder)


file_db = toolbox.read_folder_as_database(
    search_folder=analysis_files_folder, 
    pattern = "**/*.mat",
    columns=["Condition", "Subject", "Structure", "Date"])

plots_db=file_db.merge(pd.DataFrame([["lfp"], ["mu"]], columns=["signal"]), how="cross")

plots_db["x_data_col"] = "x_"+ plots_db["filename"] + "_"+ plots_db["signal"]
plots_db["y_data_col"] = "y_"+ plots_db["filename"] + "_"+ plots_db["signal"]

print(plots_db)

def result_computer(path, sr, signal):
    return np.arange(0, 100), np.random.rand(100)

data = {
    axis+"_"+ row["filename"] + "_"+ row["signal"]:arr
    for _,row in plots_db.iterrows()
    for axis, arr in zip(["x", "y"], result_computer(row["path"], 25000, row["signal"])) 
}

toolbox.add_draw_metadata(plots_db, 
                        fig_group=["Condition"],
                        row_group=["Condition", "signal"],
                        col_group=["Structure"],
                        # color_group=["Condition"],
)

plotcanvas = toolbox.prepare_figures(plots_db)

plotcanvas.plot(plots_db.iloc[0:2,: ], data, linestyle='dashed', marker='o')
plt.show()