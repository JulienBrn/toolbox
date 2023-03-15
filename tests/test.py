import pathlib
import pandas as pd
import logging
import beautifullogger
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
# import pickle
# import matplotlib
# import joblib
import toolbox
# import seaborn as sns

logger=logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
# matplotlib.use("tkagg")

res_folder = pathlib.Path(sys.argv[0]).parent / "Results"
analysis_files_folder = pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MonkeyData4Review/")
cache_folder= pathlib.Path(sys.argv[0]).parent / "cache"

# memory = joblib.Memory(cache_folder)


file_db = toolbox.read_folder_as_database(
    search_folder=analysis_files_folder, 
    pattern = "**/*.mat",
    columns=["Date", "Structure", "Subject", "Condition", "Unit"])

plots_db=file_db.merge(pd.DataFrame([["lfp"], ["mu"]], columns=["signal"]), how="cross")

print(plots_db)