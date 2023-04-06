from toolbox import Manager, np_loader, df_loader, float_loader, matlab_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import beautifullogger, pathlib, pandas as pd, toolbox, numpy as np
from tqdm import tqdm

beautifullogger.setup()
tqdm.pandas(desc="computing apply")
m = Manager("./tests/cache")

cols = ["Condition", "Subject", "Structure", "Date"]
df_handle = m.declare_computable_ressource(
    read_folder_as_database, {
      "search_folder": pathlib.Path("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review"),
      "columns":cols,
       "pattern": "**/*.mat"}, df_loader, "input_file_df", save=True)

df = df_handle.get_result()
print(df.columns)
df["input"] = df.apply(lambda row: m.declare_ressource(row["path"], matlab_loader, "raw", id="_".join(row[cols].to_list()+[row["filename"]])), axis=1)
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

def get_columns(df, columns):
    if isinstance(columns, str):
        columns=[columns]
    def compute_and_clean_row(row):
        dres={}
        for col in columns:
          res = row[col].get_result()
          if hasattr(res, "shape"):
            dres[col] = "Shape{}".format(res.shape)
          elif hasattr(res, "__len__"):
            dres[col] = "{}_of_{}_elements".format(type(res), len(res))
          else:
             dres[col] = res
          row[col].save()
        for col in row.index:
            if isinstance(row[col], toolbox.RessourceHandle) and not col in columns:
                row[col].unload()
        return dres
    df[columns] = df.progress_apply(compute_and_clean_row, axis=1, result_type="expand")
    print(df.columns)

get_columns(df, ["clean_bounds", "lfp_sig", "lfp_fs"])
print(df.columns)
result_df = df[[col for col in df.columns if not isinstance(df[col].iat[0], toolbox.RessourceHandle)]]
print(result_df.columns)

df_loader.save("final.tsv", result_df)