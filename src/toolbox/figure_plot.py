import pandas as pd, numpy as np, functools, xarray as xr, dask.dataframe as dd, dask
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
from dask.diagnostics import ProgressBar
import graphchain

logger = logging.getLogger(__name__)

def transform_kwarg(x, data):
    if hasattr(x, "__iter__") and not isinstance(x, str) and len(x)<10:
        cols = []
        for col in x:
            if not col in data.columns and not col in data.index.names:
                return x
            else:
                cols.append(col)
        data[", ".join(cols)] = data[cols].apply(lambda cols: ", ".join([str(c) for c in cols]), axis=1)

        return ", ".join(cols)
    else:
        return x


class FigurePlot:
    def __init__(self, data, *, figures=None, fig_title: str ="", subplot_title = "{default}", **kwargs):
        data = data.reset_index().copy()
        nkwarg = {k:transform_kwarg(v, data) for k,v in kwargs.items()}
        from string import Formatter
        title_keys = [t for t in Formatter().parse(fig_title)]
        if figures is None:
            figures = [str(t[1]) for t in title_keys if not t[1] is None]
        if len(figures) == 0 :
            data["__fig_str"] = ""
            figures=["__fig_str"]
        fig_groups = data.groupby(by=figures, observed=True) 
        self.figures = {}
        self.groups = fig_groups
        for t,d in fig_groups:
            facetgrid = sns.FacetGrid(d, **nkwarg)
            self.figures[t] = facetgrid
            df_value_dict = d.apply(lambda x: x.iat[0]).to_dict()
            facet_data = facetgrid.facet_data()
            for i, row_name in enumerate(facetgrid.row_names):
                for j, col_name in enumerate(facetgrid.col_names):
                    ax = facetgrid.axes[i, j]
                    default_title = ax.title._text
                    default_value_dict = {"default": "" if "margin_titles" in kwargs and kwargs["margin_titles"] else default_title, "row_name": row_name, "col_name": col_name}
                    (i1,j1,k1), fd = next(facet_data) 
                    if i1 ==i and j1 ==j:
                        subplot_value_dict = fd.apply(lambda x: x.iat[0]).to_dict()
                    else:
                        raise Exception(f"Subplotfacet data error {i1} {i} {j1} {j}")
                    all_dict= dict(**default_value_dict, **subplot_value_dict)
                    newtitle = subplot_title.format(**{t[1]: all_dict[t[1]] for t in Formatter().parse(subplot_title)})
                    if "margin_titles" in kwargs and kwargs["margin_titles"]:
                        if default_title:
                            newtitle = f"{default_title}\n{newtitle}"
                    ax.set_title(newtitle)
            if fig_title !="":
                try:
                    facetgrid.figure.suptitle(fig_title.format(**df_value_dict))
                    facetgrid.figure.subplots_adjust(top=.9, bottom=0.05)
                    facetgrid.figure.canvas.manager.set_window_title(fig_title.format(**df_value_dict))
                except:
                    print(df_value_dict)
                    print(fig_title)
                    raise
        self.data=data

    def map(self, func, *args, **kwargs):
        for facetgrid in self.figures.values():
            facetgrid.map_dataframe(func, *args, **kwargs)
        return self
    
    def add_legend(self, *args, **kwargs):
        for facetgrid in self.figures.values():
            facetgrid.add_legend(*args, **kwargs)
        return self
    def tight_layout(self, *args, **kwargs):
        for facetgrid in self.figures.values():
            facetgrid.tight_layout(*args, **kwargs)
        return self
    
    def maximize(self):
        for facetgrid in self.figures.values():
            facetgrid.figure.set_figwidth(19)
            facetgrid.figure.set_figheight(10)
            facetgrid.figure.canvas.manager.window.showMaximized()
        return self
        
    def save_pdf(self, pdf_writer_or_path):
        from matplotlib.backends.backend_pdf import PdfPages
        if not isinstance(pdf_writer_or_path, PdfPages):
            pdf_writer_or_path = PdfPages(pdf_writer_or_path)
        for facetgrid in self.figures.values():
            pdf_writer_or_path.savefig(facetgrid.figure)
        return self
    
    def refline(self, *args, **kwargs):
        for facetgrid in self.figures.values():
            facetgrid.refline(*args, **kwargs)
        return self
    
    def pcolormesh(self, *args, **kwargs):
        def colormesh(data: pd.DataFrame, x: str, y:str, value: str, ysort=None, xlabels=True, ylabels=True, **kwargs):
           data = data.set_index([x, y], append=False)
           if data.index.duplicated(keep=False).any():
               raise Exception(f"Duplication Examples:{data.index[data.index.duplicated(keep=False)]}")
           data = data[value].unstack(x)
           if not ysort is None:
                try:
                    data = data.sort_values(ysort)
                except:
                    if isinstance(ysort, float):
                        col = data.columns[np.argmin(np.abs(np.array(data.columns)-ysort))]
                        # print(f"Sorting by {col}")
                        data = data.sort_values(col)
                    else:
                        raise
           kwargs.pop("color")
        #    print(kwargs)
        #    input()
        #    if no_labels:
           plt.pcolormesh(data.to_numpy(), **kwargs)
           ax = plt.gca()
           if xlabels:
            xticks = np.linspace(0, len(data.columns)-1, 5, endpoint=True).round().astype(int)
            xtickslabels = [str(x)[0:5] for x in data.columns[xticks]]
            ax.set_xticks(xticks.astype(float)+0.5, xtickslabels)
           if ylabels:
            yticks = np.linspace(0, len(data.index)-1, 5, endpoint=True).round().astype(int)
            ytickslabels = [str(y)[0:5] for y in data.index[yticks]]
            ax.set_yticks(yticks.astype(float)+0.5, ytickslabels)
        #    else:
        #         x_coords = np.array(data.columns.to_list() + [4000])
        #         y_coords = np.array(data.index.to_list() + [51])
        #         x_coords_f = np.stack([x_coords]*len(y_coords))
        #         y_coords_f = np.stack([y_coords]*len(x_coords), axis=-1)
        #         #    print(data.shape, x_coords_f.shape, y_coords_f.shape)
        #         plt.pcolormesh(x_coords_f, y_coords_f, data.to_numpy(), **kwargs)

        for facetgrid in self.figures.values():
            facetgrid.map_dataframe(colormesh, **kwargs)

        