from __future__ import annotations
from typing import List, Tuple, Dict, Any, Union, Callable
from toolbox import json_loader, RessourceHandle, Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
from importlib.resources import files as package_data
import os

logger=logging.getLogger(__name__)

import sys, time
import matplotlib, importlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie
from PyQt5.QtCore import pyqtSlot, QItemSelectionModel, QModelIndex
from  toolbox import DataFrameModel
from toolbox.main_window_ui import Ui_MainWindow
from PyQt5.uic import loadUi
import PyQt5.QtCore as QtCore
from multiprocessing import Process
from threading import Thread
import tkinter
from PyQt5.QtCore import QThread, pyqtSignal
from toolbox.mplwidget import MplCanvas, MplWidget

import PyQt5.QtGui as QtGui
import traceback
import tqdm

def export_fig(folder, fig, title:str, canvas):
    import re, unicodedata
    canvas.draw()
    value = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    ntitle = re.sub(r'[-\s]+', '-', value).strip('-_')
    fig.savefig(str(pathlib.Path(folder) / (ntitle +".png")))

class MyResultWidget(QtWidgets.QWidget):
   def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self.export = lambda x: print("No export defined for this tab")

def mk_result_tab(nrows, ncols, **kwargs):
   tab = MyResultWidget()
   gridLayout = QtWidgets.QGridLayout(tab)
   mpls = np.empty((nrows, ncols))
   mpls=mpls.astype(object)
   for row in range(nrows):
      for col in range(ncols):
         widget_3 = QtWidgets.QWidget(tab)
         verticalLayout_7 = QtWidgets.QVBoxLayout(widget_3)
         mpl = MplWidget(widget_3)
         mpl.canvas.ax = mpl.canvas.fig.subplots(1,1, subplot_kw=kwargs)
         verticalLayout_7.addWidget(mpl)
         toolbar = NavigationToolbar(mpl.canvas, parent=widget_3)
         gridLayout.setColumnStretch(col, 1)
         gridLayout.setRowStretch(row, 1)
         gridLayout.addWidget(widget_3, row, col)
         mpls[row, col] = mpl
   return tab, mpls

   # self.label_3 = QtWidgets.QLabel(self.widget_3)
   # self.label_3.setObjectName("label_3")
   # self.verticalLayout_7.addWidget(self.label_3)
   # self.label_4 = QtWidgets.QLabel(self.widget_3)
   # self.label_4.setObjectName("label_4")
   # self.verticalLayout_7.addWidget(self.label_4)
   # self.gridLayout.addWidget(self.widget_3, 0, 2, 1, 2)

# def mk_result_tab():
#    result_tab = QtWidgets.QWidget()
#    verticalLayout_4 = QtWidgets.QVBoxLayout(result_tab)
#    mpl = MplWidget(result_tab)
#    verticalLayout_4.addWidget(mpl)
#    toolbar = NavigationToolbar(mpl.canvas, parent=result_tab)
#    return result_tab, mpl

# class GetResult(QThread):
#    progress = pyqtSignal(int)
#    def __init__(self, model, indices, cols):
#       super().__init__()
#       self.model = model
#       self.indices = indices
#       self.cols = cols
#    def run(self):
#       model = self.model
#       indices = self.indices
#       df = model._dataframe
#       tqdm.pandas(desc="Getting results") 
#       i=0
#       nb_done_since=0
#       last_time = time.time()
#       for index in tqdm(indices):
#          for colind, col in enumerate(self.cols):
#             if isinstance(df[col].iat[index], toolbox.RessourceHandle):
#                df[col].iat[index].get_result()
#                # model.dataChanged.emit(
#                #    model.createIndex(index,colind),  model.createIndex(index+1,colind+1), (QtCore.Qt.EditRole,)
#                # ) 
#          i=i+1
#          nb_done_since=nb_done_since+1
#          curr_time = time.time()
#          if curr_time - last_time > 0.5:
#             self.progress.emit(nb_done_since)
#             model.dataChanged.emit(model.createIndex(0,0), model.createIndex(len(model._dataframe.index), len(model._dataframe.columns)), (QtCore.Qt.EditRole,)) 
#             nb_done_since=0
#             last_time =curr_time
#       self.progress.emit(nb_done_since)

# class GetDataframe(QThread):
#    dfcomputed = pyqtSignal(pd.DataFrame)
#    def __init__(self, df):
#       super().__init__()
#       self.df = df
#    def run(self):
#     try:
#       res = self.df.get_df().reset_index(drop=True)
#       self.dfcomputed.emit(res)
#     except:
#       logger.error("Impossible to get dataframe")
#       self.dfcomputed.emit(pd.DataFrame([], columns=["Error"]))

# class ViewResult(QThread):
#    ready = pyqtSignal(int)
#    def __init__(self, df, result_tabs, rows):
#       super().__init__()
#       self.df = df
#       if hasattr(df, "get_nb_figs"):
#          nb_figs = df.get_nb_figs(rows)
#       else:
#          nb_figs = 1
#       self.canvas=[]
#       for i in range(nb_figs):
#          result_tab, mpl = mk_result_tab()
#          self.canvas.append(mpl.canvas)
#          result_tabs.addTab(result_tab, "res"+str(i))
#       self.rows = rows
#    def run(self):
#       if hasattr(self.df, "show_figs"):
#          for l in self.df.show_figs(self.rows, self.canvas):
#             logger.info("Emitting {}".format(l))
#             for i in l:
#                self.ready.emit(i)
#       else:
#          canvas = self.canvas[0]
#          if len(self.rows.index) == 1 or not hasattr(self.df, "view_items"):
#             self.df.view_item(canvas, self.rows.iloc[0, :])
#          else:
#             self.df.view_items(canvas, self.rows)
#          self.ready.emit(0)
# from datetime import datetime

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie, QCursor
from PyQt5.QtCore import pyqtSlot, QItemSelectionModel, QModelIndex

      


class GUIDataFrame:
   name: str
   metadata: Dict[str, str]
   tqdm: tqdm.tqdm
   _dataframe: toolbox.RessourceHandle[pd.DataFrame]

   def __init__(self, name,metadata: Dict[str, str], df_ressource_manager: toolbox.Manager, other_dfs={}, save=False, alternative_names =[]):
      #Metadata param should be handled by func
      self.name = name
      self.alternative_names =alternative_names
      self.metadata = metadata
      # for df in other_dfs.values():
      #    self.metadata = {**self.metadata, **df.metadata}
      self.tqdm = tqdm.tqdm
      self.df_ressource_manager = df_ressource_manager
      self.other_dfs=other_dfs
      self.save = save
      self.update_df()

   def get_df(self) -> pd.DataFrame: 
      self.update_df()
      return self._dataframe_out.get_result()
      
   def get_full_df(self) -> pd.DataFrame:
      self.update_df()
      return self._dataframe.get_result()
   
   def update_df(self):
      def mcompute_df(**kwargs):
         logger.info("Computing df {}".format(self.name))
         for t in self.other_dfs.values():
            t.tqdm = self.tqdm
         ret= self.compute_df(**kwargs)
         if hasattr(self, "key_columns"):
            ret = ret[self.key_columns + [col for col in ret.columns if not col in  self.key_columns]]
         logger.info("Done Computing df {}".format(self.name))
         return ret
      
      def mcompute_output_df(ret):
         dropped_cols = [col for col in ret.columns if col[0]=='_']
         if "_Discarded" in ret.columns:
            return ret[ret["_Discarded"]!=True].reset_index(drop=True).drop(columns=dropped_cols)
         else:
            return ret.drop(columns=dropped_cols)
      
      for df in self.other_dfs.values():
         df.update_df()
      df_params = {n:df._dataframe_out for n, df in self.other_dfs.items()}
      meta_params = {n.replace(".", "_"):v for n,v in self.metadata.items()}
      old_id = self._dataframe.id if hasattr(self, "_dataframe") else None
      self._dataframe = self.df_ressource_manager.declare_computable_ressource(mcompute_df,dict(**df_params, **meta_params), df_loader, "Dataframe {}".format(self.name), self.save)
      self._dataframe_out = self.df_ressource_manager.declare_computable_ressource(mcompute_output_df, {"ret":self._dataframe}, df_loader, "Dataframe out {}".format(self.name), False)
      updated = self._dataframe.id != old_id
      if updated:
         logger.info(f"Updated dataframe {self.name} Metadata now is\n{self.metadata}")
      return updated
      # return self._dataframe.get_result()
   
   def compute_df(self, **kwargs):
      raise NotImplementedError("Abstract method compute_df of GUIDataFrame")

class Task(QtCore.QObject):
   name: str
   status: Union["Pending", "Running", "Cancelled", "Aborted", "Finished"]

   onstart: Callable #Guaranteed to be called from GUI Thread
   onend: Callable #Guaranteed to be called from GUI Thread
   run: Callable #Usually called in separate thread
   kwargs: Dict[str, Any]

   errors: List[Any]
   warnings: List[Any]
   progress: tqdm.tqdm

   update_signal = pyqtSignal(float, float, str)

   process: TaskThread

   def __init__(self, window, name, onstart, onend, run, kwargs, on_curr_thread=False):
      super().__init__()
      self.window = window
      self.name = name
      self.status="Pending"
      self.onstart = onstart
      self.onend= onend
      self.f = run
      self.kwargs = kwargs
      self.errors = []
      self.warnings = []

      self.update_signal.connect(lambda cur, total, display: self.update_bar(cur, total, display))
      self.process=None
      self.on_curr_thread = on_curr_thread

   def update_bar(self, cur, total, display_str):
      progress_bar = self.window.progressBar
      progress_bar.setMaximum(int(total))
      progress_bar.setValue(int(cur))
      progress_bar.setFormat(display_str)

   def run(self):
      from toolbox.patch_tqdm import patch_tqdm
      progress_class = patch_tqdm(self.update_signal)
      self.window.progressBar.show()
      self.status = "Running"
      def finish():
         self.onend(**self.kwargs, task_info={"errors": self.errors, "warnings": self.warnings, "progress": progress_class, "result":self.process.res})
         self.status = "Finished"
         self.window.process_task()
      try:
         if not self.onstart(**self.kwargs, task_info={"errors": self.errors, "warnings": self.warnings, "progress": progress_class}) is False:
            if not self.on_curr_thread:
               self.process = TaskThread(self.f, self.kwargs, self.errors, self.warnings, progress_class)
               self.process.finished.connect(finish)
               self.process.start()
            else:
               r = self.f(**self.kwargs, task_info={"errors": self.errors, "warnings": self.warnings, "progress": progress_class})
               self.process = TaskThread(self.f, self.kwargs, self.errors, self.warnings, progress_class)
               self.process.res = r
               finish()
      except KeyboardInterrupt as e:
         raise e
      except BaseException as e:
         tb = traceback.format_exc()
         logger.error("error while computing: {}. Traceback:\n{}".format(e, tb))
         self.errors.append(e)

   def abort(self):
      if not self.process is None:
         self.process.terminate()

      


class TaskThread(QThread):
   def __init__(self, f, kw, err, warn, progress):
      super().__init__()
      self.f =f
      self.kw = kw
      self.err = err
      self.warn= warn
      self.progress = progress

   def run(self):
      try:
         r = self.f(**self.kw, task_info={"errors": self.err, "warnings": self.warn, "progress": self.progress})
         self.res = r
      except KeyboardInterrupt as e:
         raise e
      except BaseException as e:
         logger.error("error while computing: {}".format(e))
         self.err.append(e)


class Window(QMainWindow, Ui_MainWindow):
   setup_ready = pyqtSignal(dict)
   dfs: List[Any]
   tasks: List[Task]
   task_num: int
   current_df: int

   tableView: QTableView

   def __init__(self, parent=None):
      super().__init__(parent)
      self.setupUi(self)
      self.dfs = []
      self.tasks = []
      self.task_num = 0
      self.reload_dfs_list_view()
      self.reload_setup_params_view()
      self.initialize_nice_gui()
      self.initialize_events()
      self.process_task()
      self.current_df = None

   

   def add_task(self, t: Task):
         self.tasks.append(t)
         self.process_task()
         # self.current_exec.setText(t.name)
         # logger.info("running task {}".format(t.name))
         # t.run()
         # logger.info("done running task {}".format(t.name))
         # self.current_exec.setText(t.name)
   
   def process_task(self, abort_cur=False):
      while self.task_num < len(self.tasks) and not self.tasks[self.task_num].status in ["Pending", "Running"]:
         self.task_num+=1

      if abort_cur:
         self.tasks[self.task_num].abort()
         self.task_num+=1

      if self.task_num >= len(self.tasks):
         self.current_exec.setText("All done")
         self.progressBar.hide()
         return
      else:
         t = self.tasks[self.task_num]
         nb_unprocessed = len(self.tasks) - self.task_num
         self.current_exec.setText("{} running, {} waiting".format(1, nb_unprocessed-1))
         # t.progress = self.progressBar #TOREMOVE
         if not t.status =="Running":
            nb_unprocessed = len(self.tasks) - self.task_num
            self.current_exec.setText("{} running, {} waiting".format(1, nb_unprocessed-1))
            t.run()

      

   def initialize_events(self):
      # self.invalidate.clicked.connect(lambda: self.add_task(self.mk_invalidate_task([i.row() for i in self.tableView.selectionModel().selectedRows()])))
      import itertools
      self.compute.clicked.connect(
         lambda:self.add_task(self.mk_compute_task(
            itertools.product(
               range(len(self.dfs[self.current_df].get_df().index)), 
               range(len(self.dfs[self.current_df].get_df().columns)
            )), [(toolbox.RessourceHandle.is_saved_on_disk, False), (toolbox.RessourceHandle.is_saved_on_compute, True)]
         )))
      self.view.clicked.connect(lambda: self.view_all_bis())
      self.exportall.clicked.connect(lambda: self.export_all_figures())
      self.export_btn.clicked.connect(self.save_df_file_dialog)
      # self.next.clicked.connect(get_next)
      # self.previous.clicked.connect(get_prev)
      self.aborttask.clicked.connect(lambda: self.process_task(abort_cur=True))
      self.menu_tabs.currentChanged.connect(lambda index: self.reload_from_selection() if index==1 else None)
      self.dataframe_list.clicked.connect(lambda index: self.reload_from_selection())
      self.result_tabs.tabCloseRequested.connect(lambda i: self.tab_close(i))

   def initialize_nice_gui(self):
      self.tableView.setSortingEnabled(True)
      self.splitter.setStretchFactor(1,6)
      self.setup_params_tree.header().setDefaultSectionSize(250)
      self.setup_params_tree.expandAll()
      sp_retain =self.progressBar.sizePolicy()
      sp_retain.setRetainSizeWhenHidden(True)
      self.progressBar.setSizePolicy(sp_retain)
      self.dataframe_list.expandAll()
      self.result_tabs.clear()
      # for t in self.result_tabs.children()[0].children():
      #    print(t.objectName())
      # input()

   def add_df(self, df):
      self.dfs.append(df)
      self.reload_dfs_list_view()
      self.reload_setup_params_view()
      self.dataframe_list.expandAll()

   #### HANDLING OF DF LIST OBJECT ################
   def reload_dfs_list_view(self):
      df = pd.DataFrame([[df.name] for df in self.dfs], columns=["Name"])
      self.dataframe_list.setModel(toolbox.DictTreeModel(df))

   def reload_from_selection(self):
      selected_indices = self.dataframe_list.selectionModel().selectedRows()
      if len(selected_indices) ==1:
         curr_index = selected_indices[0]
         full_name=[]
         while curr_index.isValid():
            full_name.append(curr_index.data())
            curr_index = curr_index.parent()
         full_name.reverse()
         full_name = ".".join(full_name)
         try:
            self.current_df = [i for i, df in enumerate(self.dfs) if df.name == full_name or full_name in df.alternative_names][0]
         except:

            return
         self.update_from_setup_params_view()

         def computefunc(df, current_df, task_info):
               for d in self.dfs:
                  d.tqdm = task_info["progress"]
               return df.get_full_df()

         def displayfunc(df, current_df, task_info):
            ndf = task_info["result"]
            if self.current_df == current_df:
               try:
                  self.tableView.setModel(DataFrameModel(ndf.reset_index(drop=True)))
                  if self.tableView.model().rowCount() < 500:
                     self.tableView.resizeColumnsToContents()
               except BaseException as e:
                  display = [str(e), str(traceback.format_exc())]
                  if isinstance(ndf, toolbox.Error):
                     string = str(ndf)
                     for s in string.split("\n"):
                        display.append(s)
                  self.tableView.setModel(DataFrameModel(pd.DataFrame([[e] for e in display], columns=["Error"])))
                  self.tableView.resizeColumnsToContents()
                  self.tableView.resizeRowsToContents()
               # self.tableView.resizeColumnsToContents()

         # if self.dfs[self.current_df].update_df():
         self.add_task(Task(self, "compute df {}".format(self.dfs[self.current_df].name), 
                              lambda df, current_df, task_info: True, displayfunc, computefunc, 
                              {"df":self.dfs[self.current_df], "current_df": self.current_df}))

   def select_df(self, index):
      name = self.dfs[index].name
      l = name.split(".")
      search = self.dataframe_list.model().item_dict
      for n in l:
         search = search[2][n]
         self.dataframe_list.setExpanded(search[0].index(), True)
      mindex = search[0].index()
      self.dataframe_list.selectionModel().select(mindex, QItemSelectionModel.Select)
      self.reload_from_selection()

   

   #### HANDLING OF SETUP_TREE_VIEW OBJECT ################
   def reload_setup_params_view(self):
      d = {}
      for df in self.dfs:
         d = {**d, **df.metadata}
      df = pd.DataFrame(d.items(), columns=['Parameter Name', 'Parameter Value'])
      self.setup_params_tree.setModel(toolbox.DictTreeModel(df))
      self.setup_params_tree.expandAll()
    
   def update_from_setup_params_view(self):
      df: pd.DataFrame = self.setup_params_tree.model().get_values().set_index('Parameter Name')
      setup_params = df['Parameter Value'].to_dict()
      self.set_setup_params(setup_params, update_view=False)

   def set_setup_params(self, setup_params, update_view=True):
      for i in range(len(self.dfs)):
         if {k:v for k, v in setup_params.items() if k in self.dfs[i].metadata} != self.dfs[i].metadata:
            self.dfs[i].metadata = {k:v for k, v in setup_params.items() if k in self.dfs[i].metadata}
            # self.dfs[i].invalidated=True
            # self.dfs[i]._dataframe.invalidate_all()
      if update_view:
         self.reload_setup_params_view()

   ##### HANDLING OF TASKS ###############

   ##### HANDLING OF EVENTS ##############

   def mk_invalidate_task(self, indices):
      def invalidate(df, indices, __curr_task):
         for i in tqdm(indices):
            for col in df.result_columns:
               val = df[col].iat[i]
               if isinstance(val, RessourceHandle):
                  val.invalidate_all()
      
      return Task("invalidate", len(indices), invalidate, [], {"df": self.tableView.model()._dataframe, "indices":indices})
   
   def mk_compute_task(self, indices, filters=[]):
      df = self.tableView.model()._dataframe
      def mfilter(item: toolbox.RessourceHandle):
         if isinstance(item, toolbox.RessourceHandle):
            if item.is_in_memory():
               return False
            for f,val in filters:
               if not f(item) is val:
                  return False
            return True
         return False
      
      def update():
         self.tableView.model().dataChanged.emit(
         self.tableView.model().createIndex(0, 0), 
         self.tableView.model().createIndex(1, 1))
         pass
      def run(task_info):
          nonlocal indices
          indices = list(indices)
          mtqdm:tqdm.tqdm = task_info["progress"]
          items: List[toolbox.RessourceHandle] = [df.iloc[i, j] for i, j in mtqdm(indices, desc="Preparing compute") if mfilter(df.iloc[i, j])]
          last_time=time.time()
          for item in mtqdm(items):
              item.get_result()
              curr_time = time.time()
              if curr_time - last_time > 5:
                 last_time=curr_time
                 update()
      task = Task(self, "compute", lambda task_info: True, lambda task_info: update(), run, {}, on_curr_thread=False)
      return task
         

   def tab_close(self, i):
      self.result_tabs.removeTab(i)

   def on_computation_tab_clicked(self):
      if self.current_df is None:
         if len(self.dfs) == 0:logger.warning("No dfs")
         else:
            pass



   def view_row(self, row):
      df = self.dfs[self.current_df]
      if hasattr(df, "view") or hasattr(df, "view_bis"):
         # self.result_tabs.clear()
         self.menu_tabs.setCurrentWidget(self.result_tab)
         if hasattr(df, "view_bis"):
            df.view_bis(row, self.result_tabs)
         else:
            result_tab,mpls = mk_result_tab(1,1)
            self.result_tabs.addTab(result_tab, "res")
            df.view(row, mpls[0,0].canvas.ax, mpls[0,0].canvas.fig)
            mpls[0,0].canvas.draw()

   def view_all(self):
      df = self.dfs[self.current_df]
      if hasattr(df, "view_all"):
         self.result_tabs.clear()
         self.menu_tabs.setCurrentWidget(self.result_tab)
         result_tab,mpls = mk_result_tab(1,1)
         self.result_tabs.addTab(result_tab, "res")
         df.view_all(mpls[0,0].canvas.ax, mpls[0,0].canvas.fig)
         mpls[0,0].canvas.draw()

   def view_all_bis(self):
      df = self.dfs[self.current_df]
      if hasattr(df, "view_all_bis"):
         # self.result_tabs.clear()
         self.menu_tabs.setCurrentWidget(self.result_tab)
         df.view_all_bis(self.result_tabs)
         self.curr_view_all = self.dfs[self.current_df]

   def filter_view(self, indices):
      return indices
      # if hasattr(self.dfs[self.current_df], "time_series"):
      #    d = self.dfs[self.current_df].time_series
      #    l = [(i, j) for i,j in indices if self.tableView.model()._dataframe.columns[j] in d]
      #    return 

   def mk_view_task(self, indices):
      self.curr_view_all = False
      df = self.tableView.model()._dataframe
      indices = self.filter_view(indices)
      # d = self.dfs[self.current_df].time_series
      # nindices = [(i, k) for i,j in indices if self.tableView.model()._dataframe.columns[j] in d for k in [j, j+1]]
      nindices = indices
      t = self.mk_compute_task(nindices)
      oldend = t.onend
      def new_end(task_info):
         oldend(task_info)
         mtqdm:tqdm.tqdm = task_info["progress"]
         arrays = [toolbox.get(df.iloc[i, j]) for i, j in mtqdm(indices, desc="Gathering arrays")]
         arrays = [item for item in arrays if hasattr(item, "size") and item.size > 10]
         
         d = pd.DataFrame([[i %3] for i, _ in enumerate(arrays)], columns=["fig"])
         d["y_data"] = arrays
         # d["fig"] = df.index % 2
         d["row"] = ((d.index/3)%3).astype(int)
         d["column"] = ((d.index/9)%3).astype(int)
         d["color"] = (d.index/27).astype(int)
         self.result_tabs.clear()
         

         full_mpls = []
         for i in range(d["fig"].max()+1):
            result_tab,mpls = mk_result_tab(int(d.loc[d["fig"]==i, "row"].max()+1), int(d.loc[d["fig"]==i, "column"].max()+1))
            self.result_tabs.addTab(result_tab, "res"+str(i))
            full_mpls.append(mpls)

         def draw(f, i, j, data):
            try:
               full_mpls[f][i, j].canvas.ax.plot(data)
               full_mpls[f][i, j].canvas.draw()
            except BaseException as e:
               logger.error("{}".format(e))

         d.apply(lambda row: draw(int(row["fig"]), row["row"], row["column"], row["y_data"]), axis=1)

      def run(self):
         if hasattr(self.df, "show_figs"):
            for l in self.df.show_figs(self.rows, self.canvas):
               logger.info("Emitting {}".format(l))
               for i in l:
                  self.ready.emit(i)
         else:
            canvas = self.canvas[0]
            if len(self.rows.index) == 1 or not hasattr(self.df, "view_items"):
               self.df.view_item(canvas, self.rows.iloc[0, :])
            else:
               self.df.view_items(canvas, self.rows)
            self.ready.emit(0)


      t.onend = new_end
      return t
   
   def export_all_figures(self):
      # if self.curr_view_all:
      dir = QFileDialog.getExistingDirectory(self, "Select folder to export to")
      if dir:
         try:
            ind=0
            for i,b in enumerate(self.result_tabs.children()):
               self.result_tabs.setCurrentWidget(b)
               for j, t in enumerate(b.children()):
                  if hasattr(t, "export"):
                     self.result_tabs.setCurrentIndex(ind)

                     logger.info(f"Export defined for tab {i}.{j} {b.objectName()}.{t.objectName()} of type {type(b)}.{type(t)}")
                     t.export(str(dir))
                     ind+=1
                  else:
                     logger.info(f"No export defined for tab {i}.{j} {b.objectName()}.{t.objectName()} of type {type(b)}.{type(t)}")
         except BaseException as e:
            logger.error(e)


   #  def __init__(self, parent=None):
   #    super().__init__(parent)
   #    self.setupUi(self)
   #    self.dfs = []
   #    self.df_listmodel = QStandardItemModel()
   #    self.listView.setModel(self.df_listmodel)
   #    self.df_models = []
   #    self.curr_df = None
   #    self.process= None
   #    self.tableView.setSortingEnabled(True)
   #    self.splitter.setStretchFactor(1,6)
   #    self.progressBar.setValue(0)
   #    #   self.toolbar = NavigationToolbar(self.mpl.canvas, parent=self.result_tab)
   #    #   self.result_tabs.setTabsClosable(True)

   #    self.setup_model = QStandardItemModel()
   #    self.setup_model.setHorizontalHeaderLabels(['Parameter Name', 'Parameter Value'])
   #    self.treeView.setModel(self.setup_model)
   #    self.treeView.header().setDefaultSectionSize(250)
   #    self.setup_params=[self.setup_model.invisibleRootItem(), None, {}]


   #    def compute(indices):
   #       if self.curr_df is None:
   #          logger.error("Nothing to compute")
   #       elif self.process and self.process.isRunning():
   #          logger.error("A computational process is already running, please wait")
   #       else:
   #          self.progressBar.setMaximum(len(indices))
   #          self.progressBar.setValue(0)
   #          def progress(amount):
   #             self.progressBar.setValue(self.progressBar.value()+amount)

   #          self.process = GetResult(self.tableView.model(), indices, self.dfs[self.curr_df].result_columns)
   #          self.process.progress.connect(progress)
   #          self.process.start()
          
   #    def view(indices):
   #       # if not self.mpl.canvas.ax is None:
   #       #   if hasattr(self.mpl.canvas.ax, "flat"):
   #       #     for ax in self.mpl.canvas.ax.flat:
   #       #       ax.remove()
   #       #   else: 
   #       #       self.mpl.canvas.ax.remove()
   #       #   self.mpl.canvas.draw()
         
   #       self.result_tabs.clear()
   #       #  result_tab, mpl = mk_result_tab()
   #       #  self.result_tabs.addTab(result_tab, "res")
   #       # #  self.toolbar.setParent(None)
   #       #  self.mpl.reset()
   #       #  self.toolbar = NavigationToolbar(self.mpl.canvas, parent=self.result_tab)
   #       if hasattr(self.dfs[self.curr_df], "view_params"):
   #          params = {}
   #          def rec_print(root, prefix):
   #             if root[2] == {}:
   #                params[prefix+root[0].text()] = root[1].text()
   #             for child, val in root[2].items():
   #                rec_print(val, prefix+root[0].text()+".")
   #          for child, val in self.view_params_dict[2].items():
   #             rec_print(val, "")
   #          self.dfs[self.curr_df].view_params = params
         
   #       self.process = ViewResult(self.dfs[self.curr_df], self.result_tabs, self.tableView.model()._dataframe.iloc[indices, :])
   #       self.tabWidget.setCurrentWidget(self.result_tab)
   #       self.loader_label.setVisible(True)
   #       def when_ready(i):
   #          self.process.canvas[i].draw()
   #          #   self.toolbar.update()
   #          if i == len(self.process.canvas)-1:
   #             self.loader_label.setVisible(False)
   #             self.figs =  [canvas.fig for canvas in self.process.canvas]
   #       self.process.ready.connect(when_ready)
   #       self.process.start()
   #          # self.process.run()
   #          # when_ready()

   #    def invalidate(indices):
   #       for i in tqdm(indices):
   #          for col in self.dfs[self.curr_df].result_columns:
   #             val = self.tableView.model()._dataframe[col].iat[i]
   #             if isinstance(val, RessourceHandle):
   #                val.invalidate_all()
   #       self.tableView.model().dataChanged.emit(
   #          self.tableView.model().createIndex(0,0), self.tableView.model().createIndex(len(self.tableView.model()._dataframe.index), len(self.tableView.model()._dataframe.columns)), (QtCore.Qt.EditRole,)
   #       ) 
   #    def get_next():
   #       current_position = self.tableView.selectionModel().selectedRows()
   #       if current_position == []:
   #          current_position = -1
   #       else:
   #          current_position = current_position[0].row()
   #       query_str = str(self.query.text())
   #       if query_str == "":
   #          return
   #       self.tableView.clearSelection()
   #       #   logger.info("curr_pos = {}, query = {}".format(current_position, query_str))
   #       def verifies_query(row):
   #          items = query_str.split(",")
   #          row_str = " ".join([str(v) for v in row.values])
   #          for it in items:
   #             if not it in row_str:
   #                return False
   #          return True
   #       positions = self.tableView.model()._dataframe.apply(verifies_query, axis=1)
   #       new_pos = self.tableView.model()._dataframe[(positions) & (self.tableView.model()._dataframe.index > current_position)]
   #       logger.info(new_pos)
   #       if len(new_pos.index) ==0:
   #          new_pos = self.tableView.model()._dataframe[(positions)]
   #       if len(new_pos.index) >0:
   #          logger.info("moving to {}".format(new_pos.index[0]))
   #          self.tableView.scrollTo(self.tableView.model().createIndex(new_pos.index[0],0))
   #          select = QtCore.QItemSelection()
   #          for i in new_pos.index:
   #             select.select(self.tableView.model().createIndex(i,0), self.tableView.model().createIndex(i,len(self.tableView.model()._dataframe.columns)))
   #          self.tableView.selectionModel().select(select , QtCore.QItemSelectionModel.Select)
   #          self.tableView.model().dataChanged.emit(
   #          self.tableView.model().createIndex(0,0), self.tableView.model().createIndex(len(self.tableView.model()._dataframe.index), len(self.tableView.model()._dataframe.columns)), (QtCore.Qt.EditRole,)
   #          ) 
   #    def get_prev():
   #       current_position = self.tableView.selectionModel().selectedRows()
   #       if current_position == []:
   #          current_position = len(self.tableView.model()._dataframe.index)
   #       else:
   #          current_position = current_position[len(current_position)-1].row()
   #       query_str = str(self.query.text())
   #       if query_str == "":
   #          return
   #       self.tableView.clearSelection()
   #    #   logger.info("curr_pos = {}, query = {}".format(current_position, query_str))
   #       def verifies_query(row):
   #          items = query_str.split(",")
   #          row_str = " ".join([str(v) for v in row.values])
   #          for it in items:
   #             if not it in row_str:
   #                return False
   #          return True
   #       positions = self.tableView.model()._dataframe.apply(verifies_query, axis=1)
   #       new_pos = self.tableView.model()._dataframe[(positions) & (self.tableView.model()._dataframe.index < current_position)]
   #       logger.info(new_pos)
   #       if len(new_pos.index) ==0:
   #          new_pos = self.tableView.model()._dataframe[(positions)]
   #       if len(new_pos.index) >0:
   #          logger.info("moving to {}".format(new_pos.index[0]))
   #          self.tableView.scrollTo(self.tableView.model().createIndex(new_pos.index[0],0))
   #          select = QtCore.QItemSelection()
   #          for i in new_pos.index:
   #             select.select(self.tableView.model().createIndex(i,0), self.tableView.model().createIndex(i,len(self.tableView.model()._dataframe.columns)))
   #          self.tableView.selectionModel().select(select , QtCore.QItemSelectionModel.Select)
   #          self.tableView.model().dataChanged.emit(
   #             self.tableView.model().createIndex(0,0), self.tableView.model().createIndex(len(self.tableView.model()._dataframe.index), len(self.tableView.model()._dataframe.columns)), (QtCore.Qt.EditRole,)
   #          ) 

   #    self.invalidate.clicked.connect(lambda: invalidate([i.row() for i in self.tableView.selectionModel().selectedRows()]))
   #    self.compute.clicked.connect(lambda: compute([i.row() for i in self.tableView.selectionModel().selectedRows()]))
   #    self.view.clicked.connect(lambda: view([i.row() for i in self.tableView.selectionModel().selectedRows()]))
   #    self.exportall.clicked.connect(lambda: self.export_all_figures())
   #    self.export_btn.clicked.connect(self.save_df_file_dialog)
   #    self.next.clicked.connect(get_next)
   #    self.previous.clicked.connect(get_prev)
   #    self.tabWidget.currentChanged.connect(lambda index: self.on_computation_tab_clicked() if index==1 else None)

        

   #    def load_config():
   #       path, ok = QFileDialog.getOpenFileName(self, caption="Setup parameters to load from", filter="*.json")
   #       try:
   #          self.set_setup_params(json_loader.load(path))
   #       except:
   #          logger.error("Impossible to load configuration file")
   #    self.load_params.clicked.connect(load_config)

   #    def export_config():
   #       path, ok = QFileDialog.getSaveFileName(self, caption="Save setup parameters to", filter="*.json")
   #       try:
   #          json_loader.save(path, self.get_setup_params())
   #       except BaseException as e:
   #          logger.error("Impossible to save configuration :{}\n".format(e))
   #    self.export_params.clicked.connect(export_config)




   #    self.loader_label = QtWidgets.QLabel(self.centralwidget)
   #    # self.label.setGeometry(QtCore.QRect(0, 0, 0, 0))
      
   #    self.loader_label.setMinimumSize(QtCore.QSize(250, 250))
   #    self.loader_label.setMaximumSize(QtCore.QSize(250, 250))

   #    # Loading the GIF
   #    self.movie = QMovie(str(package_data("analysisGUI.ui").joinpath("loader.gif")))
   #    self.loader_label.setMovie(self.movie)

   #    self.movie.start()
   #    self.loader_label.setVisible(False)

   #  def resizeEvent(self, event):
   #        #  super(Ui_MainWindow, self).resizeEvent(event)
   #         super(QMainWindow, self).resizeEvent(event)
   #         self.move_loader()

   #  def move_loader(self):
   #    self.loader_label.move(int(self.size().width()/2), int(self.size().height()/2)-125)
   #    pass


   #  def add_df(self, df, switch = False):
   #      self.dfs.append(df)
   #      self.df_models.append(None)#DataFrameModel(df.get_df().reset_index(drop=True))
   #      self.df_listmodel.appendRow(QStandardItem(df.name))

   #      params = df.metadata
   #      for p, val in params.items():
   #         keys = p.split(".")
   #         curr = self.setup_params
   #         for k in keys:
   #            if not k in curr[2]:
   #              curr[0].appendRow([QStandardItem(k), QStandardItem("")])
   #              if not curr[1] is None:
   #                curr[1].setEditable(False)
   #              curr[0].child(curr[0].rowCount() - 1, 1).setEditable(True)
   #              curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
   #              curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), curr[0].child(curr[0].rowCount() - 1, 1), {}]
   #            curr = curr[2][k]
   #         curr[1].setText(str(val))
   #      self.treeView.expandAll()

   #      if switch:
   #        self.on_listView_clicked(self.listView.model().index(len(self.dfs) -1, 0))
          
   #        # self.curr_df = len(self.dfs) -1
   #        # self.tableView.setModel(self.df_models[self.curr_df])
       
   #  def on_computation_tab_clicked(self):
   #    setup_params = self.get_setup_params()
   #    self.setup_ready.emit(setup_params)
   #    for i in range(len(self.df_models)):
   #      if {k:v for k, v in setup_params.items() if k in self.dfs[i].metadata} != self.dfs[i].metadata:
   #        self.dfs[i].metadata = {k:v for k, v in self.get_setup_params().items() if k in self.dfs[i].metadata}
   #        self.dfs[i].invalidated=True


   #    if self.curr_df is None:
   #      self.curr_df = 0
   #    self.on_listView_clicked(self.listView.model().index(self.curr_df, 0))

   #  def set_setup_params(self, params):
   #    for p, val in params.items():
   #        keys = p.split(".")
   #        curr = self.setup_params
   #        for k in keys:
   #          if not k in curr[2]:
   #            curr[0].appendRow([QStandardItem(k), QStandardItem("")])
   #            if not curr[1] is None:
   #              curr[1].setEditable(False)
   #            curr[0].child(curr[0].rowCount() - 1, 1).setEditable(True)
   #            curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
   #            curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), curr[0].child(curr[0].rowCount() - 1, 1), {}]
   #          curr = curr[2][k]
   #        curr[1].setText(str(val))
   #    self.treeView.expandAll()
       
   #  def get_setup_params(self):
   #    params = {}
   #    def rec_print(root, prefix):
   #      if root[2] == {}:
   #         params[prefix+root[0].text()] = root[1].text()
   #      for child, val in root[2].items():
   #          rec_print(val, prefix+root[0].text()+".")
   #    for child, val in self.setup_params[2].items():
   #      rec_print(val, "")
   #    return params
        
   def save_df_file_dialog(self):
      filename, ok = QFileDialog.getSaveFileName(
         self,
         "Select file to export to", 
         filter = "(*.tsv)"
      )
      if filename:
         to_save_df = self.dfs[self.current_df].get_df()
         # to_save_df["coherence_pow_path"] = to_save_df.apply(lambda row: str(row["coherence_pow"].get_disk_path()), axis=1)
         # to_save_df["coherence_pow_core_path"] = to_save_df.apply(lambda row: str(row["coherence_pow"].manager.d[row["coherence_pow"].id]._core_path), axis=1)
         df_loader.save(filename, to_save_df)

   #  def export_all_figures(self):
   #     dir = QFileDialog.getExistingDirectory(self, "Select folder to export to")
   #     if dir:
   #        for i, fig in enumerate(self.figs):
   #           fig.savefig(pathlib.Path(dir) / "figure_{}.png".format(i), dpi=200)
   #        logger.info("exported")


   #  @QtCore.pyqtSlot("QModelIndex")
   #  def on_listView_clicked(self, model_index):
   #     self.listView.setCurrentIndex(model_index)
   #     self.curr_df = model_index.row()
   #     if self.dfs[self.curr_df].invalidated:
   #        # self.dfs[self.curr_df].metadata = {k:v for k, v in self.get_setup_params().items() if k in self.dfs[self.curr_df].metadata}
   #        self.loader_label.setVisible(True)
   #        self.process = GetDataframe(self.dfs[self.curr_df])
   #        def dataframe_ready(df):
   #           self.df_models[self.curr_df] = DataFrameModel(df)
   #           self.tableView.setModel(self.df_models[self.curr_df])
   #           self.loader_label.setVisible(False)
   #        self.process.dfcomputed.connect(dataframe_ready)
   #        self.process.start()
   #     else:
   #        self.df_models[self.curr_df] = DataFrameModel(self.dfs[self.curr_df].get_df())
   #        self.tableView.setModel(self.df_models[self.curr_df])
   #        self.tableView.setModel(self.df_models[self.curr_df])

   #     self.view_params_model = QStandardItemModel()
   #     self.view_params_model.setHorizontalHeaderLabels(['Parameter Name', 'Parameter Value'])
   #     self.view_params.setModel(self.view_params_model)
   #     self.view_params.header().setDefaultSectionSize(120)
   #     self.view_params_dict=[self.view_params_model.invisibleRootItem(), None, {}]

   #     view_params = self.dfs[self.curr_df].view_params if hasattr(self.dfs[self.curr_df], "view_params") else {}
   #     for p, val in view_params.items():
   #        keys = p.split(".")
   #        curr = self.view_params_dict
   #        for k in keys:
   #          if not k in curr[2]:
   #            curr[0].appendRow([QStandardItem(k), QStandardItem("")])
   #            if not curr[1] is None:
   #              curr[1].setEditable(False)
   #            curr[0].child(curr[0].rowCount() - 1, 1).setEditable(True)
   #            curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
   #            curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), curr[0].child(curr[0].rowCount() - 1, 1), {}]
   #          curr = curr[2][k]
   #        curr[1].setText(str(val))
   #     self.view_params.expandAll()


