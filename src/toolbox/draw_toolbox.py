import pathlib
import pandas as pd
import logging
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib as mpl

logger=logging.getLogger()

def add_draw_metadata(
        metadata, 
        fig_group=[], 
        row_group=[], 
        col_group=[], 
        color_group=[]):
    
    fig_groups = metadata.groupby(by=fig_group).groups if fig_group != [] else {"":metadata.index}
    figs=[]
    for fi, (fn, fentries) in enumerate(fig_groups.items()):
        
        fmetadata = metadata.loc[fentries, :]
        row_groups = fmetadata.groupby(by=row_group).groups if row_group != [] else {"":metadata.index}
        col_groups = fmetadata.groupby(by=col_group).groups if col_group != [] else {"":metadata.index}
        
        for ri, (rn, rentries) in enumerate(row_groups.items()):
            for ci, (cn, centries) in enumerate(col_groups.items()):
                sub_metadata = fmetadata.loc[(rentries.intersection(centries)), :]
                color_groups = sub_metadata.groupby(by=color_group).groups if color_group != [] else {"":metadata.index}
                nb_colors=len(color_groups)
                colors=plt.cm.get_cmap()
                for colori, (colorn, colorentries) in enumerate(color_groups.items()):
                    select=fentries.intersection(rentries).intersection(centries).intersection(colorentries)
                    if fig_group != []:
                        metadata.loc[select, "Figure"] = fi
                        metadata.loc[select, "Figure_label"] = str(fn)
                    if row_group != []:
                        metadata.loc[select, "Row"] = ri
                        metadata.loc[select, "Row_label"] = str(rn)
                    if col_group !=[]:
                        metadata.loc[select, "Column"] = ci
                        metadata.loc[select, "Column_label"] = str(cn)
                    if color_group!= []:
                        metadata.loc[select, "Color"] = mpl.colors.rgb2hex(colors(float(colori)/nb_colors), keep_alpha=True)
                        metadata.loc[select, "Color_label"] = str(colorn)


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