import pathlib
import pandas as pd
import logging
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

logger=logging.getLogger()

def add_draw_metadata(
        metadata, 
        fig_group=[], 
        row_group=[], 
        col_group=[], 
        color_group=[]):
    
    reserved_columns = [
        "Figure", "Figure_label", 
        "Row", "Row_label", 
        "Column", "Column_label", 
        "Color", "Color_label"
    ]

    if set(reserved_columns) & set(metadata.columns) != set():
        logger.warning("Metadata already has columns reserved for drawing.\n"
                       +"Used columns are: {}". format(set(reserved_columns) & set(metadata.columns)))

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
                colors=plt.cm.get_cmap("hsv")
                for colori, (colorn, colorentries) in enumerate(color_groups.items()):
                    select=fentries.intersection(rentries).intersection(centries).intersection(colorentries)
                    def mk_label(keys, val):
                        # logger.info("keys:\n{}\nval:\n{}".format(keys,val))
                        def label(k, v):
                            if isinstance(v, str):
                                return v
                            elif isinstance(v, float):
                                return "{}={:.2f}".format(k, v)
                            elif len(str(v)) < 15:
                                return "{}={}".format(k, str(v))
                            else:
                                return "{}={:15s}".format(k, str(v))
                        if isinstance(val, str):
                            val=[val]
                        res = ", ".join([label(k, v) for k, v in zip(keys, val)])
                        # logger.info("res={}\n".format(res))
                        return res
                    if fig_group != []:
                        metadata.loc[select, "Figure"] = fi
                        metadata.loc[select, "Figure_label"] = mk_label(fig_group, fn)
                    if row_group != []:
                        metadata.loc[select, "Row"] = ri
                        metadata.loc[select, "Row_label"] = mk_label(row_group, rn)
                    if col_group !=[]:
                        metadata.loc[select, "Column"] = ci
                        metadata.loc[select, "Column_label"] = mk_label(col_group, cn)
                    if color_group!= []:
                        metadata.loc[select, "Color"] = mpl.colors.rgb2hex(colors(float(colori)/nb_colors), keep_alpha=True)
                        metadata.loc[select, "Color_label"] = mk_label(color_group, colorn)
                    


def prepare_figures(metadata, **kwargs):
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
        ax=f.subplots(nb_rows, nb_cols, squeeze=False, subplot_kw=kwargs)
        for ri in range(nb_rows):
            for ci in range(nb_cols):
                select = (metadata["Figure"] == i) & (metadata["Row"] == ri) & (metadata["Column"] == ci)
                if "Color_label" in metadata.columns:
                    colorl = metadata.loc[select, "Color_label"].unique().tolist()
                    colorv = metadata.loc[select, "Color"].unique().tolist()
                    handles=[plt.plot([], color=colorv, label=colorl)[0] for colorv, colorl in zip(colorv,colorl)]
                    ax[ri, ci].legend(handles=handles,)
            add_headers(f, 
                row_headers=metadata.loc[metadata["Figure"] == i, "Row_label"].unique() if "Row_label" in metadata.columns else None,
                col_headers=metadata.loc[metadata["Figure"] == i, "Column_label"].unique() if "Column_label" in metadata.columns else None,
            )

        figs.append(f)
        axes.append(ax)
    return PlotCanvas(figs, axes)


def prepare_figures2(metadata, figs_param, ignore_legend = False, **kwargs):
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
    logger.info("number of figs provided {}, number of figs required {}".format(len(figs_param), nb_figs))
    for i in range(nb_figs):
        f = figs_param[i]
        if "Figure_label" in metadata.columns:
            f.suptitle("Figure {}, num_plots {}".format(
                metadata.loc[metadata["Figure"] == i, "Figure_label"].iloc[0],
                len(metadata.loc[metadata["Figure"] == i]))
            )
        nb_rows = int(metadata.loc[metadata["Figure"] == i, "Row"].max())+1
        nb_cols = int(metadata.loc[metadata["Figure"] == i, "Column"].max())+1
        ax=f.subplots(nb_rows, nb_cols, squeeze=False, subplot_kw=kwargs)
        for ri in range(nb_rows):
            for ci in range(nb_cols):
                select = (metadata["Figure"] == i) & (metadata["Row"] == ri) & (metadata["Column"] == ci)
                if "Color_label" in metadata.columns:
                    color_nums = metadata.loc[select, ["Color", "Color_label"]].groupby(by=["Color"])["Color_label"].agg(["count", "first"])
                    # print(color_nums)
                    # colorl = metadata.loc[select, "Color_label"].unique().tolist()
                    # colorv = metadata.loc[select, "Color"].unique().tolist()
                    color_nums["final_label"] = color_nums.apply(lambda row: row["first"] if row["count"]==1 else "{} (n={})".format(row["first"], row["count"]), axis=1)
                    handles=[ax[ri, ci].plot([], color=colorv, label=colorl)[0] for colorv, colorl in zip(color_nums.index.to_list(),color_nums["final_label"].to_list())]
                    if not ignore_legend :
                        ax[ri, ci].legend(handles=handles, loc='upper right', fancybox=True, shadow=True, framealpha=0.1)
                    else:
                        ax[ri, ci].legend(handles=handles, loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(0.75, 1.), framealpha=0.1)
        # handles, labels = ax[nb_rows-1, nb_cols-1].get_legend_handles_labels()
        # f.legend(handles=handles,labels=labels, loc='lower center', fancybox=True, shadow=True)
        if "Row_label" in metadata.columns:
            row_headers=[]
            for ri in range(nb_rows):
                row_headers.append(metadata.loc[(metadata["Figure"] == i) & (metadata["Row"] == ri), "Row_label"].iat[0])
        if "Column_label" in metadata.columns:
            col_headers=[]
            for ci in range(nb_cols):
                col_headers.append(metadata.loc[(metadata["Figure"] == i) & (metadata["Column"] == ci), "Column_label"].iat[0])
        # add_headers(f, 
        #     row_headers=metadata.loc[metadata["Figure"] == i, "Row_label"].unique() if "Row_label" in metadata.columns else None,
        #     col_headers=metadata.loc[metadata["Figure"] == i, "Column_label"].unique() if "Column_label" in metadata.columns else None,
        # )
        add_headers(f, 
            row_headers=row_headers if "Row_label" in metadata.columns else None,
            col_headers=col_headers if "Column_label" in metadata.columns else None,
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

    def plot(self, metadata, data, x="x_data_col", y= "y_data_col", **kwargs):
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
            ax.plot(data[row[x]], data[row[y]], color=row["Color"], label="_index: "+str(row_index), **kwargs)

    def plot2(self, metadata, x="x_data_col", y= "y_data_col", use_zscore = False, **kwargs):
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
            x_data = row[x].get_result() if hasattr(row[x], "get_result") else row[x]
            y_data = row[y].get_result() if hasattr(row[y], "get_result") else row[y]
            if y_data is np.nan or x_data is np.nan:
                logger.error("Impossible to plot data")
                continue
            # logger.info("x_data is:\n{}".format(x_data))
            try:
                if use_zscore:
                    ax.plot(x_data, scipy.stats.zscore(y_data), color=row["Color"], label="_index: "+str(row_index), **kwargs)
                else:
                    if hasattr(x_data, "shape") or hasattr(x_data, "__len__"):
                        ax.plot(x_data, y_data, color=row["Color"], label="_index: "+str(row_index), **kwargs)
                    else:
                        ax.scatter([x_data], [y_data], color=row["Color"], label="_index: "+str(row_index), **kwargs)
            except:
                logger.error("Impossible to plot data:\n{}".format(row))


    def scatter_plot2(self, metadata, x="x_data_col", y= "y_data_col", use_zscore = False, **kwargs):
        metadata=metadata.copy()
        for axs in self.axes:
            for ax in axs.reshape(-1):
                ax.set_ylim([0.01,1])
                ax.set_yscale('log')
                ax.set(xticklabels=[])
                ax.set_rlabel_position(-90)
                ax.set_yticks([0.03, 0.1, 0.3, 1], labels=[str(v) for v in [0.03, 0.1, 0.3, 1]])
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
            x_data = row[x].get_result() if hasattr(row[x], "get_result") else row[x]
            y_data = row[y].get_result() if hasattr(row[y], "get_result") else row[y]
            if y_data is np.nan or x_data is np.nan:
                logger.error("Impossible to plot data")
                continue
            # logger.info("x_data is:\n{}".format(x_data))
            try:
                if use_zscore:
                    ax.plot(x_data, scipy.stats.zscore(y_data), color=row["Color"], label="_index: "+str(row_index), **kwargs)
                else:
                    # if hasattr(x_data, "shape") or hasattr(x_data, "__len__"):
                    #     ax.plot(x_data, y_data, color=row["Color"], label="_index: "+str(row_index), **kwargs)
                    # else:
                        ax.scatter([x_data], [y_data], color=row["Color"], label="_index: "+str(row_index), **kwargs)
            except:
                logger.error("Impossible to plot data:\n{}".format(row))

    def scatter_plot_vlines(self, metadata, x="x_data_col", y= "y_data_col", use_zscore = False, **kwargs):
        metadata=metadata.copy()
        for axs in self.axes:
            for ax in axs.reshape(-1):
                ax.set_ylim([0.01,1])
                
                ax.set_yscale('log')
                # ax.set_xticks([0, -90], labels=[str(v)+"°" for v in [0, -90]])
                ax.set_xticks([0,-np.pi/2], labels=["0°", "-90°"])
                ax.set_xlim([0,2*np.pi])
                ax.set_rlabel_position(-90)
                ax.set_yticks([0.03, 0.1, 0.3, 1], labels=[str(v) for v in [0.03, 0.1, 0.3, 1]])
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
            x_data = row[x].get_result() if hasattr(row[x], "get_result") else row[x]
            y_data = row[y].get_result() if hasattr(row[y], "get_result") else row[y]
            if y_data is np.nan or x_data is np.nan:
                logger.error("Impossible to plot data")
                continue
            # logger.info("x_data is:\n{}".format(x_data))
            try:
                if use_zscore:
                    ax.plot(x_data, scipy.stats.zscore(y_data), color=row["Color"], label="_index: "+str(row_index), **kwargs)
                else:
                    # if hasattr(x_data, "shape") or hasattr(x_data, "__len__"):
                    #     ax.plot(x_data, y_data, color=row["Color"], label="_index: "+str(row_index), **kwargs)
                    # else:
                    # logger.info(x_data.size)
                    ax.vlines(x_data*np.pi/180.0, np.zeros(x_data.size)+0.01, y_data, color=row["Color"], label="_index: "+str(row_index), **kwargs)
            except:
                logger.error("Impossible to plot data:\n{}".format(row))
    
    def pcolormesh(self, metadata, data, **kwargs):
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
            ax.pcolormesh(data[row["x_data_col"]], data[row["y_data_col"]], data[row["c_data_col"]], label="_index: "+str(row_index), **kwargs)


def draw_data(data, metadata):
    logger.warning("This feature will be removed soon. Use prepare_figures and call .plot on result instead")
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