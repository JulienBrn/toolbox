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
beautifullogger.setup(displayLevel=logging.INFO)
matplotlib.use("tkagg")

res_folder = pathlib.Path(sys.argv[0]).parent / "Results"
analysis_files_folder = pathlib.Path("/media/usert4/Seagate Expansion Drive1/VictorLeroy/New_videos/New_setup")
cache_folder= pathlib.Path(sys.argv[0]).parent / "cache"

# memory = joblib.Memory(cache_folder)


file_db = toolbox.read_folder_as_database(
    search_folder=analysis_files_folder,
    pattern = "**/*.MP4",
    columns=["Type_manip", "Condition", "Sub_condition"])

print(file_db)