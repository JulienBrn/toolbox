from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy
from tqdm import tqdm

beautifullogger.setup(displayLevel=logging.WARNING)
tqdm.pandas(desc="computing apply")
m = Manager("./tests/cache")

cols = ["Condition", "Subject", "Structure", "Date"]
df_handle = m.declare_computable_ressource(
    read_folder_as_database, {
      "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"),
      "columns":cols,
       "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

df = df_handle.get_result()
df["Species"] = "Monkey"
print(df.columns)
df["sample_id"] = df.apply(lambda row: "_".join(row[cols].to_list()+[row["filename"]]), axis=1)
df["input"] = df.apply(lambda row: m.declare_ressource(row["path"], matlab_loader, "raw", id=row["sample_id"]), axis=1)
df = mk_block(df, ["input"], lambda input: input["RAW"][0,], (np_loader, "raw", False), m)
df["raw_fs"] = 25000
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


signal_reshape = pd.DataFrame({
  "signal": {"lfp": "lfp_sig", "mu": "mu_sig"},  
  "signal_fs": {"lfp": "lfp_fs", "mu": "mu_fs"},  
})

df_signal = toolbox.dataframe_reshape(df, "signal_type", signal_reshape)

df_signal["welch_window_duration"] = 3

def pwelch(signal, signal_fs, welch_window_duration):
   return scipy.signal.welch(signal, signal_fs, nperseg=welch_window_duration*signal_fs)

df_signal = mk_block(df_signal, ["signal", "signal_fs", "welch_window_duration"], pwelch, 
                     {0: (np_loader, "welch_f", True), 1: (np_loader, "welch_pow", True)}, m)

df_pair_of_group = toolbox.group_and_combine(df, ["Condition", "Subject", "Structure", "Species"])

result_df = toolbox.get_columns(df_signal, ["signal", "signal_fs", "welch_pow", "welch_f"])
print(result_df)
df_loader.save("result.tsv", result_df)