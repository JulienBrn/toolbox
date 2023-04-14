from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast
from tqdm import tqdm

import cProfile
import pstats
from pstats import SortKey

beautifullogger.setup(logmode="w")
logger=logging.getLogger(__name__)
logging.getLogger("toolbox.ressource_manager").setLevel(logging.WARNING)
logging.getLogger("toolbox.signal_analysis_toolbox").setLevel(logging.WARNING)

computation_m = Manager("./tests/cache/computation")
folder_manager = Manager("./tests/cache/folder_contents")
dataframe_manager = Manager("./tests/cache/dataframes")



####INPUT Dataframe#######

INPUT_Columns = ["Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel", "signal_type", "signal_fs", "file_path", "file_keys"]

def mk_monkey_input() -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"),
         "columns": ["Condition", "Subject", "Structure", "Date"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)
   df: pd.DataFrame = df_handle.get_result()
   df["Species"] = "Monkey"
   df["Session"] = "MS_#"+ df.groupby(by=["Date", "filename", "Subject"]).ngroup().astype(str)
   df["SubSessionInfo"] = 0
   df["SubSessionInfoType"] = "order"
   df["Channel"] = df["filename"]
   
   def get_monkey_signals(row: pd.Series) -> pd.DataFrame:
      row_raw = row.copy()
      row_raw["signal_type"] = "raw"
      row_raw["signal_fs"] = 25000
      row_raw["file_path"] = row["path"]
      row_raw["file_keys"] = ("RAW", (0,))

      row_spikes = row.copy()
      row_spikes["signal_type"] = "spike_times"
      row_spikes["signal_fs"] = 25000
      row_spikes["file_path"] = row["path"]
      row_spikes["file_keys"] = ("SUA", (0,))

      res = pd.DataFrame([row_raw, row_spikes])
      return res

   tqdm.pandas(desc="Creating monkey metadata")
   return pd.concat(df.progress_apply(get_monkey_signals, axis=1).values, ignore_index=True)

def mk_human_input() -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review"),
         "columns":["Structure", "Date_HT", "Electrode_Depth"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)
   df: pd.DataFrame = df_handle.get_result()
   df["Species"] = "Human"
   df["Condition"] = "pd"
   df["Date"] = df["Date_HT"].str.slice(0, 10)
   df["Subject"] = "#"+ df.groupby("Date").ngroup().astype(str)
   df["Session"] = "HS_#"+ df.groupby(by=["Date_HT", "Electrode_Depth", "Subject"]).ngroup().astype(str)
   df["Channel"] = df["filename"] 
   df["SubSessionInfo"] = 0
   df["SubSessionInfoType"] = "order"

   def get_human_signals(row: pd.Series) -> pd.DataFrame:
      row_raw = row.copy()
      row_raw["signal_type"] = "mua"
      row_raw["signal_fs"] = 48000 if row["Date"] < "2015_01_01" else 44000
      row_raw["file_path"] = row["path"]
      row_raw["file_keys"] = ("MUA", (0,))

      row_spikes = row.copy()
      row_spikes["signal_type"] = "spike_times"
      row_spikes["signal_fs"] = 48000 if row["Date"] < "2015_01_01" else 44000
      row_spikes["file_path"] = row["path"]
      row_spikes["file_keys"] = ("SUA", (0,))

      res = pd.DataFrame([row_raw, row_spikes])
      return res
   tqdm.pandas(desc="Creating human metadata")
   return pd.concat(df.progress_apply(get_human_signals, axis=1).values, ignore_index=True)
   
def mk_rat_input() -> pd.DataFrame :
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/NicoAnalysis/Rat_Data"),
         "columns":["Condition", "Subject", "Date", "Session", "Structure"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

   df: pd.DataFrame = df_handle.get_result()
   df["Species"] = "Rat"
   session_regex = re.compile("(?P<word>[a-z]+)(?P<num>[0-9]+)", re.IGNORECASE)
   df[["Session", "SubSessionInfo"]] = df.apply(
      lambda row: ["RS_"+ str(row["Date"]) + "_"+ str(session_regex.fullmatch(row["Session"]).group("word")), session_regex.fullmatch(row["Session"]).group("num")], 
      axis=1, result_type="expand"
   )
   df["SubSessionInfoType"] = "order"

   rat_raw_regexs = [
         re.compile("(?P<sln>.*)_Probe(?P<channel>.*)"),
         re.compile("(?P<sln>.*)_(?P<channel>EEG)ipsi", re.IGNORECASE),
         re.compile("(?P<sln>.*)_ipsi(?P<channel>EEG)", re.IGNORECASE)
      ]
   rat_spikes_regexs = [
         re.compile("(?P<sln>.*)_Pr(?P<channel>[0-9]+)_(?P<neuron>.*)"),
         re.compile("(?P<sln>.*)_mPr(?P<channel>[0-9]+)_(?P<neuron>.*)"), #make better
         re.compile("(?P<sln>.*)_P(?P<channel>[0-9]+)_(?P<neuron>.*)"),  #make better
         re.compile("(?P<sln>.*)_Pr_(?P<channel>[0-9]+)_(?P<neuron>.*)"),
         re.compile("(?P<sln>.*)_Pr_(?P<channel>[0-9]+)(?P<neuron>)"),
         re.compile("(?P<sln>.*)_Pr(?P<channel>[0-9]+)(?P<neuron>)"),
         re.compile("(?P<sln>.*)_(?P<neuron>(SS)|(mSS))_Pr_(?P<channel>[0-9]+)"),
         re.compile("(?P<sln>.*)_(?P<neuron>(SS)|(mSS))_(?P<channel>STN)"),#make better
         re.compile("(?P<sln>.*)_(?P<neuron>(SS)|(mSS))_(?P<channel>STN).*"),#make better
         re.compile("(?P<sln>.*)_(?P<channel>All)_(?P<neuron>STR)"),
      ]
   def raise_print(o):
      logger.error("Exception thrown with object:\n{}".format(o))
      raise BaseException("Error")
   def get_rat_signals(row: pd.Series) -> pd.DataFrame:
      with h5py.File(row["path"], 'r') as file:
         if row["filename"] != "Units":
            channel_dict = {key:{
               "Channel": key_match.group("channel"),
               "signal_fs": int(1/file[key]["interval"][0,0])
               } for key in file.keys() for key_match in [next((v for v in [regex.fullmatch(key) for regex in rat_raw_regexs] if v), None)]}
            res = pd.DataFrame.from_dict(channel_dict, orient="index").reset_index(names="file_keys")
            res["file_keys"] = res.apply(lambda row: (row["file_keys"], "values"), axis=1)
            res["file_path"] = row["path"]
            res["signal_type"] = "raw"
         else:
            channel_dict = {key:{
               "Channel": key_match.group("channel") if key_match else raise_print(key),
               "signal_fs": 1
               } for key in file.keys() for key_match in [next((v for v in [regex.fullmatch(key) for regex in rat_spikes_regexs] if v), None)]}
            res = pd.DataFrame.from_dict(channel_dict, orient="index").reset_index(names="file_keys")
            res["file_keys"] = res.apply(lambda row: (row["file_keys"], "times"), axis=1)
            res["file_path"] = row["path"]
            res["signal_type"] = "spike_times"
         for col in row.index:
            res[col] = row[col]
         return res
   tqdm.pandas(desc="Creating rat metadata")
   return pd.concat(df.progress_apply(get_rat_signals, axis=1).values, ignore_index=True)



logger.info("Getting input_df")
logger.info("Getting monkey_df")
monkey_input = dataframe_manager.declare_computable_ressource(
      mk_monkey_input, {},
      df_loader, "monkey_input_df", True
   ).get_result().drop(columns=["path", "filename", "ext"])

logger.info("Getting human_df")
human_input = dataframe_manager.declare_computable_ressource(
      mk_human_input, {}, 
      df_loader, "human_input_df", True
   ).get_result().drop(columns=["path", "filename", "ext", "Date_HT", "Electrode_Depth"])

logger.info("Getting rat_df")
rat_input = dataframe_manager.declare_computable_ressource(
      mk_rat_input, {}, 
      df_loader, "rat_input_df", True
   ).get_result().drop(columns=["path", "filename", "ext"])

input_df = pd.concat([monkey_input, human_input, rat_input], ignore_index=True)[INPUT_Columns]
logger.info("Input df retrieved")
if True:
   subcols = [col for col in input_df.columns if col!="file_path"]
   if input_df.duplicated(subset=subcols).any():
      logger.error(
         "Duplicates in input dataframe. Duplicates are:\n{}".format(
            input_df.duplicated(subset=subcols, keep=False).sort_values(by=subcols)))
   else:
      if input_df.isnull().sum().sum() != 0:
         logger.warning("Number of null values are\n{}".format(input_df.isnull().sum()))
      else:
         logger.info("Metadata seems ok")
      

#################DECLARING RESSOURCES###################
logger.info("Declaring ressources")

# input_df = input_df.iloc[0:50,:].copy()

tqdm.pandas(desc="Declaring file ressources")

def get_file_ressource(d):
   if pathlib.Path(d["file_path"].iat[0]).stem != "Units":
      ret =  computation_m.declare_ressource(d["file_path"].iat[0], matlab_loader, check=False)
   else:
      ret =  computation_m.declare_ressource(d["file_path"].iat[0], matlab73_loader, check=False)
   return d.apply(lambda row: ret, axis=1)

# cProfile.run('input_df.groupby("file_path", group_keys=False).progress_apply(get_file_ressource)', "restats")
# p = pstats.Stats('restats')
# perf = pd.DataFrame(
#     p.getstats(),
#     columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
# )
# p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
# 

# with toolbox.Profile() as pr:
input_df["file_ressource"] = input_df.groupby("file_path", group_keys=False).progress_apply(get_file_ressource)

# perf = pr.get_results()
# perf["filename:lineno(function)"] = perf["filename:lineno(function)"].str.replace("/home/julien/miniconda3/envs/dev/lib/python3", "python", regex=False)
# perf = perf[perf["cumtime"] > 1]

# print(perf.sort_values(by=["cumtime"], ascending=False, ignore_index=True).to_string())

tqdm.pandas(desc="Declaring array ressources")

def get_array_ressource(file_ressource, file_keys):
   ktuple = ast.literal_eval(file_keys)
   res = file_ressource
   for key in ktuple:
      res = res[key]
   return res

input_df = mk_block(
   input_df, ["file_ressource", "file_keys"], get_array_ressource, 
   (np_loader, "signal", False), computation_m)

signal_cols = ["Species", "Condition", "Subject", "Date", "Session", "SubSessionInfo", "SubSessionInfoType",  "Structure", "Channel", "signal_type", "signal_fs", "signal"]
signal_df = input_df.copy()[signal_cols]

#############CLEANED SIGNALS##########

tqdm.pandas(desc="Declaring clean signals")

cleaned_df = signal_df[signal_df["signal_type"].isin(["raw", "mua"])].copy()

clean_params = {
   "deviation_factor":5,
   "min_length":0.003,
   "join_width":3,
   "shoulder_width":1,
   "recursive": True,
   "replace_type":"affine",
   "clean_version":3,
}

for key,val in clean_params.items():
   cleaned_df[key] = val

def clean(clean_version, signal, signal_fs, deviation_factor, min_length, join_width, recursive, shoulder_width):
    bounds = toolbox.compute_artefact_bounds(signal, signal_fs, deviation_factor, min_length, join_width, recursive, shoulder_width)
    return pd.DataFrame(bounds, columns=["start", "end"])

tqdm.pandas(desc="Declaring clean bounds")
cleaned_df = mk_block(cleaned_df, ["signal", "signal_fs", "deviation_factor", "min_length", "join_width", "recursive", "shoulder_width", "clean_version"], clean,
                             (df_loader, "clean_bounds", True), computation_m)


def generate_clean(signal, clean_bounds, replace_type):
  filtered= signal.copy().astype(float)
  for _,artefact in clean_bounds.iterrows():
    s = artefact["start"]
    e = artefact["end"]
    filtered[s:e] = np.nan
  if replace_type == "affine":
     return toolbox.affine_nan_replace(filtered)
  elif replace_type == "nan":
    return filtered
  else:
     raise BaseException("Invalid replace type")
  
tqdm.pandas(desc="Declaring clean signal")
cleaned_df = mk_block(cleaned_df, ["signal", "clean_bounds", "replace_type"], generate_clean, (np_loader, "cleaned_signal", False), computation_m)  

signal_append = cleaned_df.copy().drop(columns=["clean_bounds", "cleaned_signal", "signal", "signal_type"]+list(clean_params.keys()))
signal_append["signal"] = cleaned_df["cleaned_signal"]
signal_append["signal_type"] = cleaned_df["signal_type"]+"_cleaned"
signal_df = pd.concat([signal_df, signal_append], ignore_index=True).sort_values(by=signal_cols[0:-2])

################################## LFP signals ###################################

tqdm.pandas(desc="Declaring lfp signals")

lfp_df = signal_df[signal_df["signal_type"].isin(["raw_cleaned"])].copy()

lfp_params = {
   "filter_freq":200,
   "out_fs":500,
   "order":3
}

for key,val in lfp_params.items():
   lfp_df[key] = val

def extract_lfp(signal, signal_fs, filter_freq, out_fs, order):
   lfp, out_fs = toolbox.extract_lfp(signal, signal_fs, filter_freq, out_fs, order)
   return (lfp, out_fs)

lfp_df = mk_block(lfp_df, ["signal", "signal_fs", "filter_freq","out_fs", "order"], extract_lfp, 
             {0: (np_loader, "lfp_sig", True), 1: (float_loader, "lfp_fs", True)}, computation_m)

signal_append = lfp_df.copy().drop(columns=["lfp_sig", "lfp_fs", "signal", "signal_type", "signal_fs"]+list(lfp_params.keys()))
signal_append["signal"] = lfp_df["lfp_sig"]
signal_append["signal_type"] = lfp_df["signal_type"].str.replace("raw_", "lfp_", regex=False)
signal_append["signal_fs"] = lfp_df["lfp_fs"]
signal_df = pd.concat([signal_df, signal_append], ignore_index=True).sort_values(by=signal_cols[0:-2])

################################## BUA signals ###################################

tqdm.pandas(desc="Declaring bua signals")

bua_df = signal_df[signal_df["signal_type"].isin(["raw_cleaned", "mua_cleaned"])].copy()

bua_params = {
   "filter_low_freq":300,
   "filter_high_freq":6000,
   "filter_refreq":1000,
   "out_fs":2000,
   "order":3
}

for key,val in bua_params.items():
   bua_df[key] = val

def extract_bua(signal, signal_fs, filter_low_freq, filter_high_freq, filter_refreq, out_fs, order):
   bua, out_fs = toolbox.extract_mu(signal, signal_fs, filter_low_freq, filter_high_freq, filter_refreq, out_fs, order)
   return (bua, out_fs)

bua_df = mk_block(bua_df, ["signal", "signal_fs"] + list(bua_params.keys()), extract_bua, 
             {0: (np_loader, "bua_sig", True), 1: (float_loader, "bua_fs", True)}, computation_m) 


signal_append = bua_df.copy().drop(columns=["bua_sig", "bua_fs", "signal", "signal_type", "signal_fs"]+list(bua_params.keys()))
signal_append["signal"] = bua_df["bua_sig"]
signal_append["signal_type"] = bua_df["signal_type"].str.replace("raw_", "bua_").str.replace("mua_", "bua_")
signal_append["signal_fs"] = bua_df["bua_fs"]
signal_df = pd.concat([signal_df, signal_append], ignore_index=True).sort_values(by=signal_cols[0:-2])


################################## Spike signals ###################################

tqdm.pandas(desc="Declaring continuous spike signals")

spike_df = signal_df[signal_df["signal_type"].isin(["spike_times"])].copy()

spike_params ={
   "out_fs":1000
}

for key,val in spike_params.items():
   spike_df[key] = val

def make_continuous(signal: np.array, signal_fs, out_fs):
   new_size = int(signal.max()*out_fs/signal_fs)+1
   res = np.zeros(new_size)
   indexes = (signal *out_fs/signal_fs).astype(int)
   res[indexes] = 1
   return res

spike_df = mk_block(spike_df, ["signal", "signal_fs"] + list(spike_params.keys()), make_continuous, 
             (np_loader, "spike_sig", False), computation_m) 


signal_append = spike_df.copy().drop(columns=["spike_sig", "signal", "signal_type", "signal_fs"]+list(spike_params.keys()))
signal_append["signal"] = spike_df["spike_sig"]
signal_append["signal_type"] = "spike_continuous"
signal_append["signal_fs"] = spike_df["out_fs"]
signal_df = pd.concat([signal_df, signal_append], ignore_index=True).sort_values(by=signal_cols[0:-2], ignore_index=True)



######### PWELCH_ANALYSIS ###########################

tqdm.pandas(desc="Declaring pwelch results")

pwelch_df = signal_df[signal_df["signal_type"].isin(["lfp_cleaned", "bua_cleaned"])].copy()

pwelch_params ={
   "welch_window_duration":3
}
for key,val in pwelch_params.items():
   pwelch_df[key] = val

def pwelch(signal, signal_fs, welch_window_duration):
   return scipy.signal.welch(signal, signal_fs, nperseg=welch_window_duration*signal_fs)

pwelch_df = mk_block(pwelch_df, ["signal", "signal_fs", "welch_window_duration"], pwelch, 
                     {0: (np_loader, "welch_f", True), 1: (np_loader, "welch_pow", True)}, computation_m)


######### COHERENCE ANALYSIS #####################




beautifullogger.setDisplayLevel(logging.INFO)
logger.debug("Signal dataframe is:\n{}".format(signal_df.to_string()))
logger.info("Signal dataframe has {} entries. A snapshot is:\n{}".format(len(signal_df.index),signal_df.iloc[0:10,:].to_string()))
beautifullogger.setDisplayLevel(logging.DEBUG)

tqdm.pandas(desc="Computing results")
print(toolbox.get_columns(signal_df, ["signal"]))

raise BaseException("stop")











RAW_Columns = ["Species", "Condition", "Subject", "Structure", "Session", "Channel", "raw", "raw_fs"] + ["Clean_bounds"] + ["Clean_sig"] + ["lfp", "lfp_fs", "mu", "mu_fs"]
SPIKES_Columns = ["Species", "Condition", "Subject", "Structure", "Session", "Channel", "Neuron", "times", "times_fs"] + ["signal"]
Signal_Columns = ["Species", "Condition", "Subject", "Structure", "Session", "Channel", "signal_type", "Neuron", "signal", "signal_fs"] #Neuron is None with signal_type != spike
PWELCH_Columns = Signal_Columns + ["pwelch_f", "pwelch_pow"] #(possibly filtered to not do on spikes) 
Coherence_Columns = ["Species", "Condition", "Subject", "Session",
                          "channel_1", "signal_type_1", "Neuron_1", "signal_1", "signal_fs_1",
                          "channel_2", "signal_type_2", "Neuron_2", "signal_2", "signal_fs_2"
                     ] + ["coherence_f", "coherence_pow, coherence_phase"]

Correlation_Columns = ["Species", "Condition", "Subject", "Session",
                          "channel_1", "Neuron_1", "signal_1", "signal_fs_1",
                          "channel_2", "Neuron_2", "signal_2", "signal_fs_2"
                      ] + ["coherence_f", "coherence_pow, coherence_phase"] #signal_type is "spike"







def mk_monkey_metadata():
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"),
         "columns": ["Condition", "Subject", "Structure", "Date"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)
   df: pd.DataFrame = df_handle.get_result().iloc[0:5, :].copy()
   df["Species"] = "Monkey"
   df["Session"] = df["Date"] + df["filename"]
   df["Channel"] = df["filename"]
   tqdm.pandas(desc="Declaring Monkey ressources")
   df["input"] = df.progress_apply(lambda row: computation_m.declare_ressource(row["path"], matlab_loader, "raw", id=row["path"]), axis=1)
   return df

def mk_human_metadata():
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review"),
         "columns":["Structure", "Date_HT", "Electrode_Depth"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

   df = df_handle.get_result().iloc[0:5, :].copy()
   df["Species"] = "Human"
   df["Condition"] = "pd"
   df["Session"] = df["Date_HT"]+df["Electrode_Depth"] 
   df["Date"] = df["Date_HT"].str.slice(0, 10)
   df["Subject"] = df.groupby("Date").ngroup()
   df["Channel"] = df["filename"] 
   tqdm.pandas(desc="Declaring Human ressources")
   df["input"] = df.progress_apply(lambda row: computation_m.declare_ressource(row["path"], matlab_loader, "raw", id=row["path"]), axis=1)
   return df

def mk_rat_metadata():
   df_handle = folder_manager.declare_computable_ressource(
      read_folder_as_database, {
         "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/NicoAnalysis/Rat_Data"),
         "columns":["Condition", "Subject", "Date", "Session", "Structure"],
         "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

   rat_regexs = [
         re.compile("(?P<sln>.*)_Probe(?P<channel>.*)"),
         re.compile("(?P<sln>.*)_(?P<channel>EEG)ipsi", re.IGNORECASE),
         re.compile("(?P<sln>.*)_ipsi(?P<channel>EEG)", re.IGNORECASE)
      ]
   def get_rats_dataframe(prev_dataframe: pd.DataFrame):
      def get_dataframe(row):
         with h5py.File(row["path"], 'r') as file:
            key_regex=[next((v for v in [regex.fullmatch(key) for regex in rat_regexs] if v), None) for key in file.keys()]
            fs = [int(1/file[key]["interval"][0,0]) if "interval" in file[key] else np.nan for key in file.keys()]
            key_list = [[r.group(), r.group("channel"),r.group("sln")] if r else ["", ""] for r in key_regex ]
            
            metadata = [l1+ [f] for l1, f in zip(key_list, fs)]
            strange ={key:dict(file[key]) for key, (mname, chan, sln, fs) in zip(file.keys(), metadata) if chan =="" or sln=="" or fs==np.nan}
            good = {key:(mname, chan, sln, fs) for key, (mname, chan, sln, fs) in zip(file.keys(), metadata) if chan and sln and fs}
            if strange != []:
               for key, detail in strange.items():
                  logger.warning("Input {} has strange matlab structure {}. Details:\n{}".format(row["path"], key, detail))
            r = pd.DataFrame(good.values(), columns=["matlab_id", "Channel", "Subject_long_name", "raw_fs"])
            for k in row.index:
               r[k] = row[k] 
            r["Species"] = "Rat"
            r["Session"] = r["Date"].astype(str) + r["Session"]
            return r
      return pd.concat(prev_dataframe[prev_dataframe["filename"]!="Units"].apply(get_dataframe, axis=1).values, ignore_index=True)

   df = get_rats_dataframe(df_handle.get_result().iloc[0:5, :].copy())
   tqdm.pandas(desc="Declaring Rat RAW ressources")
   df["input_raw"] = df.progress_apply(lambda row: computation_m.declare_ressource(row["path"], matlab73_loader, "raw", id=row["path"]), axis=1)
   tqdm.pandas(desc="Declaring Rat Units ressources")
   df["input_units"] = df.progress_apply(
      lambda row: computation_m.declare_ressource(str(pathlib.Path(row["path"]).parent / "Units.mat"), matlab73_loader, "units") 
         if  (pathlib.Path(row["path"]).parent / "Units.mat").exists() else None, 
      axis=1)
   return df

def mk_raw_df():
   def mk_monkey_raw_df():
      df = mk_monkey_metadata()
      df["raw_fs"] = 25000
      df = mk_block(df, ["input"], lambda input: input["RAW"][0,] , (np_loader, "raw", False), computation_m)
      df.drop(columns=["Date", "path", "input", "filename", "ext"], inplace=True)
      return df

   def mk_human_raw_df():
      df = mk_human_metadata()
      df["raw_fs"] = df.apply(lambda row: 48000 if row["Date"] < "2015_01_01" else 44000, axis=1)
      df = mk_block(df, ["input"], lambda input: input["MUA"][0,] , (np_loader, "raw", False), computation_m)
      df.drop(columns=["Date_HT", "Electrode_Depth", "path", "input", "Date", "filename", "ext"], inplace=True)
      return df

   def mk_rat_raw_df():
      df = mk_rat_metadata()
      def get_raw(input_raw, matlab_id):
         return input_raw[matlab_id]["values"]
      df = mk_block(df, ["input_raw", "matlab_id"], get_raw , (np_loader, "raw", False), computation_m)
      df.drop(columns=["path", "input_raw", "input_units", "Date", "matlab_id", "Subject_long_name", "filename", "ext"], inplace=True)
      return df

   monkey_raw_df = mk_monkey_raw_df()
   human_raw_df = mk_human_raw_df()
   rat_raw_df = mk_rat_raw_df()

   raw_df = pd.concat([monkey_raw_df, human_raw_df, rat_raw_df], ignore_index=True)

   raw_df["sample_id"] = raw_df.apply(lambda row: "_".join([str(val) for val in row[["Species", "Condition", "Subject", "Structure", "Session", "Channel"]]]), axis=1)
   if not raw_df["sample_id"].is_unique:
      logger.error("Indices in RAW dataframe are not unique... Examples:\n{}".format(raw_df[raw_df["sample_id"].duplicated(keep=False)].sort_values(by=["sample_id"])))
      raise BaseException("Non unicity")
   else:
      raw_df.drop(columns=["sample_id"], inplace=True)
      logger.info("RAW df Metadata ok")
   return raw_df

raw_df = mk_raw_df()
raw_df.apply(lambda row: row["raw"].get_result(), axis=1)
# raw_df.apply(lambda row: row["raw_fs"].get_result(), axis=1)
logger.info("raw_df is\n{}".format(raw_df))

########Let us make Spikes_df##############


def mk_spikes_df():
   def mk_monkey_spikes_df():
      df = mk_monkey_metadata()
      df["Neuron"] = 1
      df["spike_fs"] = 25000
      df = mk_block(df, ["input"], lambda input: input["SUA"][0,] , (np_loader, "spike_times", False), computation_m)
      df.drop(columns=["Date", "path", "input", "filename", "ext"], inplace=True)
      return df

   def mk_human_spikes_df():
      df = mk_human_metadata()
      df["spike_fs"] = df.apply(lambda row: 48000 if row["Date"] < "2015_01_01" else 44000, axis=1)
      df["Neuron"] = 1
      df = mk_block(df, ["input"], lambda input: input["SUA"][0,] , (np_loader, "spike_times", False), computation_m)
      df.drop(columns=["Date_HT", "Electrode_Depth", "path", "input", "Date", "filename", "ext"], inplace=True)
      return df

   def mk_rat_spikes_df():
      df = mk_rat_metadata()
      df["spike_fs"] = df["raw_fs"]
      df["Neuron"] = 1
      def get_spikes(input_units, matlab_id):
         if input_units:
            logger.info("Input units:\n{}".format(dict(input_units)))
            return input_units[matlab_id]["times"]
         return np.nan
      df = mk_block(df, ["input_units", "matlab_id"], get_spikes , (np_loader, "spike_times", False), computation_m)
      df.drop(columns=["path", "input_raw", "input_units", "Date", "matlab_id", "Subject_long_name", "filename", "ext", "raw_fs"], inplace=True)
      return df
   monkey_spikes_df = mk_monkey_spikes_df()
   human_spikes_df = mk_human_spikes_df()
   rat_spikes_df = mk_rat_spikes_df()

   spikes_df = pd.concat([monkey_spikes_df, human_spikes_df, rat_spikes_df], ignore_index=True)

   spikes_df["sample_id"] = spikes_df.apply(lambda row: "_".join([str(val) for val in row[["Species", "Condition", "Subject", "Structure", "Session", "Channel", "Neuron"]]]), axis=1)
   if not spikes_df["sample_id"].is_unique:
      logger.error("Indices in SPIKES dataframe are not unique... Examples:\n{}".format(spikes_df[spikes_df["sample_id"].duplicated(keep=False)].sort_values(by=["sample_id"])))
      raise BaseException("Non unicity")
   else:
      spikes_df.drop(columns=["sample_id"], inplace=True)
      logger.info("RAW df Metadata ok")
   return spikes_df

spikes_df = mk_spikes_df()
spikes_df.apply(lambda row: row["spike_times"].get_result(), axis=1)
# spikes_df.apply(lambda row: row["spike_fs"].get_result(), axis=1)
logger.info("spikes_df is\n{}".format(spikes_df))

raise BaseException("stop")
########Let us make signal_df##############





cols = ["Species", "Condition", "Subject", "Structure", "raw_fs", "Date", "Session", "Channel"]


monkey_df_handle = m.declare_computable_ressource(
    read_folder_as_database, {
      "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"),
      "columns": ["Condition", "Subject", "Structure", "Date"],
       "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

monkey_df = monkey_df_handle.get_result()

monkey_df["Species"] = "Monkey"
# monkey_df["Electrode"] = monkey_df["filename"]
monkey_df["raw_fs"] = 25000
monkey_df["Session"] = monkey_df["Date"] + monkey_df["filename"]
monkey_df["Channel"] = monkey_df["filename"]
# monkey_df["Neuron"] = "1"

if set(cols) <= set(monkey_df.columns):
   print("Monkey df loaded")
   print(monkey_df)
else:
   raise BaseException("Not all key have been provided for monkey_df")


human_df_handle = m.declare_computable_ressource(
    read_folder_as_database, {
      "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/HumanData4review"),
      "columns":["Structure", "Date_HT", "Electrode_Depth"],
       "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

human_df = human_df_handle.get_result()
human_df["Species"] = "Human"
human_df["Condition"] = "pd"

# human_df["Site"] = human_df["Date_HT"]+human_df["Electrode_Depth"] 
human_df["Session"] = human_df["Date_HT"]+human_df["Electrode_Depth"] 
# human_df["Electrode"] = human_df.apply(lambda row: row["Electrode"] + row["Date"][10:], axis=1)
human_df["Date"] = human_df["Date_HT"].str.slice(0, 10)
human_df["Subject"] = human_df["Date"]
human_df["raw_fs"] = human_df.apply(lambda row: 48000 if row["Date"] < "2015_01_01" else 44000, axis=1)
human_df["Channel"] = human_df["filename"] 

if set(cols) <= set(human_df.columns):
   print("Human df loaded")
   print(human_df)
else:
   raise BaseException("Not all key have been provided for human_df")

print("Loading rat_df")

rat_df_tmp_handle = m.declare_computable_ressource(
    read_folder_as_database, {
      "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/NicoAnalysis/Rat_Data"),
      "columns":["Condition", "Subject", "Date", "Session", "Structure"],
       "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

import re
rat_regexs = [
      re.compile("(?P<sln>.*)_Probe(?P<channel>.*)"),
      re.compile("(?P<sln>.*)_(?P<channel>EEG)ipsi", re.IGNORECASE),
      re.compile("(?P<sln>.*)_ipsi(?P<channel>EEG)", re.IGNORECASE)
   ]
def get_rats_dataframe(prev_dataframe: pd.DataFrame):
   
   def get_dataframe(row):
      with h5py.File(row["path"], 'r') as file:
         key_regex=[next((v for v in [regex.fullmatch(key) for regex in rat_regexs] if v), None) for key in file.keys()]
         fs = [int(1/file[key]["interval"][0,0]) if "interval" in file[key] else np.nan for key in file.keys()]
         key_list = [[r.group(), r.group("channel"),r.group("sln")] if r else ["", ""] for r in key_regex ]
         
         metadata = [l1+ [f] for l1, f in zip(key_list, fs)]
         strange ={key:dict(file[key]) for key, (mname, chan, sln, fs) in zip(file.keys(), metadata) if chan =="" or sln=="" or fs==np.nan}
         good = {key:(mname, chan, sln, fs) for key, (mname, chan, sln, fs) in zip(file.keys(), metadata) if chan and sln and fs}
         if strange != []:
            for key, detail in strange.items():
               logger.warning("Input {} has strange matlab structure {}. Details:\n{}".format(row["path"], key, detail))
         r = pd.DataFrame(good.values(), columns=["Matlab_id", "Channel", "Subject_long_name", "raw_fs"])
         for k in row.index:
            r[k] = row[k] 
         r["Species"] = "Rat"
         return r
   return pd.concat(prev_dataframe[prev_dataframe["filename"]!="Units"].apply(get_dataframe, axis=1).values, ignore_index=True)

rat_df = get_rats_dataframe(rat_df_tmp_handle.get_result())

if set(cols) <= set(rat_df.columns):
   print("Rat df loaded")
   print(rat_df[[col for col in rat_df.columns if col != "path"]].to_string())
else:
   print(rat_df[[col for col in rat_df.columns if col != "path"]].to_string())
   raise BaseException("Not all key have been provided for rat_df. Missing keys are {}".format(set(cols) - set(rat_df.columns)))


df = pd.concat([monkey_df, human_df, rat_df], ignore_index=True)




#Selection
df = df[df["Species"] == "Rat"]


# print(df.columns)
df["sample_id"] = df.apply(lambda row: "_".join([str(val) for val in row[cols]]), axis=1)

if not df["sample_id"].is_unique:
   logger.error("Indices in dataframe are not unique... Examples:\n{}".format(df[df["sample_id"].duplicated(keep=False)].sort_values(by=["sample_id"])))
   raise BaseException("Non unicity")
else:
   logger.info("Metadata ok")

logger.info("Input dataframe ready with shape {}, declaring ressources...".format(df.shape))
tqdm.pandas(desc="Declaring ressources")
df.loc[df["Species"] != "Rat", "input"] = df[df["Species"] != "Rat"].progress_apply(lambda row: m.declare_ressource(row["path"], matlab_loader, "raw", id=row["sample_id"]), axis=1)
df.loc[df["Species"] == "Rat", "input"] = df[df["Species"] == "Rat"].progress_apply(lambda row: m.declare_ressource(row["path"], matlab73_loader, "raw", id=row["sample_id"]), axis=1)
logger.info("Matlab ressources declared")

def get_raw_signal(input, sample_id, Species, Matlab_id):
   if Species =="Monkey":
      if "RAW" in input:
         return input["RAW"][0,]
   if Species =="Human":
      if "MUA" in input:
         return input["MUA"][0,]
   if Species =="Rat":
      # print(input[Matlab_id]["values"].shape)
      return input[Matlab_id]["values"]
   logger.error("From file with id: {}. Impossible to retrieve data".format(sample_id, input))
   raise BaseException("From file with id: {}. Impossible to retrieve data".format(sample_id))
  

df = mk_block(df, ["input", "sample_id", "Species", "Matlab_id"], get_raw_signal , (np_loader, "raw", False), m)





df["deviation_factor"] = 5
df["min_length"] = 0.003
df["join_width"] = 3
df["shoulder_width"] = 1
df["recursive"] = True
df["replace_type"] = "affine"
df["clean_version"] = 3

df["lfp_filter_freq"] = 200
df["lfp_out_fs"] = 500
df["lfp_order"] = 3

df["mu_filter_low_freq"] = 300
df["mu_filter_high_freq"] = 6000
df["mu_filter_refreq"] = 1000
df["mu_out_fs"] = 2000
df["mu_order"] = 3

df["spike_alg"] = "given"

def clean(clean_version, raw, raw_fs, deviation_factor, min_length, join_width, recursive, shoulder_width):
    bounds = toolbox.compute_artefact_bounds(raw, raw_fs, deviation_factor, min_length, join_width, recursive, shoulder_width)
    return pd.DataFrame(bounds, columns=["start", "end"])


df = mk_block(df, ["raw", "raw_fs", "deviation_factor", "min_length", "join_width", "recursive", "shoulder_width", "clean_version"], clean,
                             (df_loader, "clean_bounds", True), m)




def generate_clean(raw, clean_bounds, replace_type):
  filtered= raw.copy().astype(float)
  for _,artefact in clean_bounds.iterrows():
    s = artefact["start"]
    e = artefact["end"]
    filtered[s:e] = np.nan
  if replace_type == "affine":
     return toolbox.affine_nan_replace(filtered)
  elif replace_type == "nan":
    return filtered
  else:
     raise BaseException("Invalid replace type")
df = mk_block(df, ["raw", "clean_bounds", "replace_type"], generate_clean, (np_loader, "cleaned_raw", False), m)                             

def extract_lfp(cleaned_raw, raw_fs, lfp_filter_freq, lfp_out_fs, lfp_order):
   lfp, out_fs = toolbox.extract_lfp(cleaned_raw, raw_fs, lfp_filter_freq, lfp_out_fs, lfp_order)
   return (lfp, out_fs)

df = mk_block(df, ["cleaned_raw", "raw_fs", "lfp_filter_freq","lfp_out_fs", "lfp_order"], extract_lfp, 
             {0: (np_loader, "lfp_sig", True), 1: (float_loader, "lfp_fs", True)}, m) 

def extract_mu(cleaned_raw, raw_fs, mu_filter_low_freq, mu_filter_high_freq, mu_filter_refreq, mu_out_fs, mu_order):
   mu, out_fs = toolbox.extract_mu(cleaned_raw, raw_fs, mu_filter_low_freq, mu_filter_high_freq, mu_filter_refreq, mu_out_fs, mu_order)
   return (mu, out_fs)

df = mk_block(df, ["cleaned_raw", "raw_fs", "mu_filter_low_freq", "mu_filter_high_freq","mu_filter_refreq", "mu_out_fs", "mu_order"], extract_mu, 
             {0: (np_loader, "mu_sig", True), 1: (float_loader, "mu_fs", True)}, m) 

def extract_spike(input, spike_alg):
   if spike_alg == "given":
      return input["SUA"][0,]
   else:
     raise BaseException("Invalid spike algorithm")

df = mk_block(df, ["input", "spike_alg"], extract_spike, (np_loader, "spike_sig", True), m) 




signal_reshape = pd.DataFrame({
  "signal": {"lfp": "lfp_sig", "mu": "mu_sig", "spike": "spike_sig"},  
  "signal_fs": {"lfp": "lfp_fs", "mu": "mu_fs", "spike": "raw_fs"},  
})

df_signal = toolbox.dataframe_reshape(df, "signal_type", signal_reshape)
df_signal = df_signal[~ ((df_signal["Species"] == "Human") &  (df_signal["signal_type"] == "lfp"))]

df_welch = df_signal.loc[df_signal["signal_type"]!="spike"].copy()

df_welch["welch_window_duration"] = 3

def pwelch(signal, signal_fs, welch_window_duration):
   return scipy.signal.welch(signal, signal_fs, nperseg=welch_window_duration*signal_fs)

df_welch = mk_block(df_welch, ["signal", "signal_fs", "welch_window_duration"], pwelch, 
                     {0: (np_loader, "welch_f", True), 1: (np_loader, "welch_pow", True)}, m)

df_pair_of_group = toolbox.group_and_combine(df, ["Condition", "Subject", "Structure", "Species"])

tqdm.pandas(desc="Computing results")

result_df = toolbox.get_columns(df_welch, ["signal", "signal_fs", "welch_pow", "welch_f"])
print(result_df)
df_loader.save("result.tsv", result_df)