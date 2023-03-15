import pathlib
import pandas as pd
import logging
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
import matplotlib as mpl

logger=logging.getLogger()


def read_folder_as_database(
        search_folder: pathlib.Path,
        columns: List[str],
        pattern: Union[str, List[str]],
):
    logger.info("Read folder as database called")
    if not isinstance(pattern, List):
        pattern = [pattern]

    raw_files=[]
    for p in pattern:
        raw_files += search_folder.glob(p)
    # logger.info("Found {} files".format(len(raw_files)))
    l=[]
    for file in raw_files:
        if file.parents[len(columns)]==search_folder:
            cols = [file.parents[i].name for i in range(len(columns))]
            cols.reverse()
            l.append([file.stem, file.suffix]+ cols + [str(file)])
        else:
            logger.debug("Ignored file {} because {} != {}".format(file, file.parents[len(columns)], search_folder))
    database = pd.DataFrame(l,columns=["filename", "ext"]+ columns +["path"])
    return database

# def get_pwelch_analysis(file, SAMPLE_RATE, target_signal, duration=3, N=None, EXIT_SAMPLE_RATE=None):
#     raw = Helper.get_raw_signal(file, N)
#     if target_signal=="lfp":
#         sig = Helper.extract_lfp(raw, SAMPLE_RATE)
#         if EXIT_SAMPLE_RATE is None:
#             EXIT_SAMPLE_RATE=1000
#     elif target_signal=="mu":
#         sig = Helper.extract_mu(raw, SAMPLE_RATE)
#         if EXIT_SAMPLE_RATE is None:
#             EXIT_SAMPLE_RATE=2500
#     else:
#         logger.error("Unknown target_signal type: {}. Current accepted values are [lfp, mu]".format(target_signal))

#     if int(SAMPLE_RATE)%int(EXIT_SAMPLE_RATE)!= 0:
#         logger.warning("Desired sample rate is not a divisor of source sample rate."+ 
#                        "Desired {}Hz, source {}Hz".format(EXIT_SAMPLE_RATE, SAMPLE_RATE))
#     sig = sig[::int(SAMPLE_RATE/EXIT_SAMPLE_RATE)]
#     #Normalizing
#     sig = (sig - sig.mean())/sig.std()
#     #pwelch
#     welch_x, welch_y  = Helper.get_welch(sig, EXIT_SAMPLE_RATE, nperseg=duration*EXIT_SAMPLE_RATE)
#     return welch_x, welch_y


def add_draw_metadata(
        metadata, 
        fig_group=[], 
        row_group=[], 
        col_group=[], 
        color_group=[],
        labels=None):
    
    fig_groups = metadata.groupby(by=fig_group).groups if fig_group != [] else {"":metadata.index}
    figs=[]
    for fi, (fn, fentries) in enumerate(fig_groups.items()):
        # f = plt.Figure()
        fmetadata = metadata.loc[fentries, :]
        row_groups = fmetadata.groupby(by=row_group).groups if row_group != [] else {"":metadata.index}
        col_groups = fmetadata.groupby(by=col_group).groups if col_group != [] else {"":metadata.index}
        # ax = f.subplots(len(row_groups.groups), len(col_groups.groups))
        for ri, (rn, rentries) in enumerate(row_groups.items()):
            for ci, (cn, centries) in enumerate(col_groups.items()):
                sub_metadata = fmetadata.loc[rentries & centries, :]
                color_groups = sub_metadata.groupby(by=color_group).groups if color_group != [] else {"":metadata.index}
                nb_colors=len(color_groups)
                colors=plt.cm.get_cmap()
                for colori, (colorn, colorentries) in enumerate(color_groups.items()):
                    if fig_group != []:
                        metadata.loc[fentries & rentries & centries & colorentries, "Figure"] = fi
                        metadata.loc[fentries & rentries & centries & colorentries, "Figure_label"] = str(fn)
                    if row_group != []:
                        metadata.loc[fentries & rentries & centries & colorentries, "Row"] = ri
                        metadata.loc[fentries & rentries & centries & colorentries, "Row_label"] = str(rn)
                    if col_group !=[]:
                        metadata.loc[fentries & rentries & centries & colorentries, "Column"] = ci
                        metadata.loc[fentries & rentries & centries & colorentries, "Column_label"] = str(cn)
                    if color_group!= []:
                        metadata.loc[fentries & rentries & centries & colorentries, "Color"] = mpl.colors.rgb2hex(colors(float(colori)/nb_colors), keep_alpha=True)
                        metadata.loc[fentries & rentries & centries & colorentries, "Color_label"] = str(colorn)


def prepare_figures(metadata):
    metadata=metadata.copy()
    if not "Figure" in metadata.columns:
        metadata["Figure"] = 0
    if not "Row" in metadata.columns:
        metadata["Row"] = 0
    if not "Column" in metadata.columns:
        metadata["Column"] = 0
    if not "Color" in metadata.columns:
        metadata["Color"] = "C0"
    nb_figs = int(metadata["Figure"].max())+1 
    figs = []
    axes= []
    for i in range(nb_figs):
        f = plt.figure(layout="tight")
        if "Figure_label" in metadata.columns:
            f.suptitle("Figure {}, num_plots {}".format(
                metadata.loc[metadata["Figure"] == i, "Figure_label"].iloc[0],
                len(metadata.loc[metadata["Figure"] == i]))
            )
        nb_rows = int(metadata.loc[metadata["Figure"] == i, "Row"].max())+1
        nb_cols = int(metadata.loc[metadata["Figure"] == i, "Column"].max())+1
        ax=f.subplots(nb_rows, nb_cols, squeeze=False)
        for ri in range(nb_rows):
            for ci in range(nb_cols):
                select = (metadata["Figure"] == i) & (metadata["Row"] == ri) & (metadata["Column"] == ci)
                if "Color_label" in metadata.columns:
                    colorl = metadata.loc[select, "Color_label"].unique().tolist()
                    colorv = metadata.loc[select, "Color"].unique().tolist()
                    handles=[plt.plot([], color=colorv, label=colorl)[0] for colorv, colorl in zip(colorv,colorl)]
                    ax[ri, ci].legend(handles=handles)
            add_headers(f, 
                row_headers=metadata.loc[metadata["Figure"] == i, "Row_label"].unique() if "Row_label" in metadata.columns else None,
                col_headers=metadata.loc[metadata["Figure"] == i, "Column_label"].unique() if "Column_label" in metadata.columns else None,
            )

        figs.append(f)
        axes.append(ax)
        return PlotCanvas(figs, axes)

class PlotCanvas:
    figs: List[plt.Figure]
    axes: List[List[plt.Axes]]

    def __init__(self, figs, axes):
        self.figs = figs
        self.axes = axes

    def plot(self, metadata, data, **kwargs):
        metadata=metadata.copy()
        if not "Figure" in metadata.columns:
            metadata["Figure"] = 0
        if not "Row" in metadata.columns:
            metadata["Row"] = 0
        if not "Column" in metadata.columns:
            metadata["Column"] = 0
        if not "Color" in metadata.columns:
            metadata["Color"] = "C0"
        for row_index,row in metadata.iterrows():
            f = self.figs[int(row["Figure"])]
            axs = self.axes[int(row["Figure"])]
            ax=axs[int(row["Row"]), int(row["Column"])]
            ax.plot(data[row["x_data_col"]], data[row["y_data_col"]], color=row["Color"], label="_index: "+str(row_index), **kwargs)


def draw_data(data, metadata):
    metadata=metadata.copy()
    logger.info(str(metadata.columns))
    if not "Figure" in metadata.columns:
        metadata["Figure"] = 0
    if not "Row" in metadata.columns:
        metadata["Row"] = 0
    if not "Column" in metadata.columns:
        metadata["Column"] = 0
    if not "Color" in metadata.columns:
        metadata["Color"] = "C0"
    nb_figs = int(metadata["Figure"].max())+1 
    figs = []
    for i in range(nb_figs):
        f = plt.figure(layout="tight")
        if "Figure_label" in metadata.columns:
            f.suptitle("Figure {}, num_plots {}".format(
                metadata.loc[metadata["Figure"] == i, "Figure_label"].iloc[0],
                len(metadata.loc[metadata["Figure"] == i]))
            )
        nb_rows = int(metadata.loc[metadata["Figure"] == i, "Row"].max())+1
        nb_cols = int(metadata.loc[metadata["Figure"] == i, "Column"].max())+1
        ax=f.subplots(nb_rows, nb_cols, squeeze=False)
        for ri in range(nb_rows):
            for ci in range(nb_cols):
                select = (metadata["Figure"] == i) & (metadata["Row"] == ri) & (metadata["Column"] == ci)
                if "Color_label" in metadata.columns:
                    colorl = metadata.loc[select, "Color_label"].unique().tolist()
                    colorv = metadata.loc[select, "Color"].unique().tolist()
                    handles=[plt.plot([], color=colorv, label=colorl)[0] for colorv, colorl in zip(colorv,colorl)]
                    ax[ri, ci].legend(handles=handles)
            add_headers(f, 
                row_headers=metadata.loc[metadata["Figure"] == i, "Row_label"].unique() if "Row_label" in metadata.columns else None,
                col_headers=metadata.loc[metadata["Figure"] == i, "Column_label"].unique() if "Column_label" in metadata.columns else None,
            )

        figs.append((f, ax))
    for row_index,row in metadata.iterrows():
        f = figs[int(row["Figure"])][0]
        axs = figs[int(row["Figure"])][1]
        ax=axs[int(row["Row"]), int(row["Column"])]
        # ax.plot(data[row["x_data_col"]], data[row["y_data_col"]], color=row["Color"], label="_index"+str(row_index))
        ax.plot(data[row["x_data_col"]], data[row["y_data_col"]], color=row["Color"])
    plt.show()


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    if row_headers is None and col_headers is None:
        return 
    # Based on/copied from  https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )