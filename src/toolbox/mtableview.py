from __future__ import annotations
from typing import List, Tuple, Dict, Any
from toolbox import json_loader, RessourceHandle, Manager, np_loader, df_loader, float_loader, matlab_loader, matlab73_loader, read_folder_as_database, mk_block, replace_artefacts_with_nans2
import toolbox
import logging, beautifullogger, pathlib, pandas as pd, toolbox, numpy as np, scipy, h5py, re, ast, sys
from tqdm import tqdm
import statsmodels.api as sm
from importlib.resources import files as package_data
import os

logger=logging.getLogger(__name__)

import sys, time
import matplotlib, importlib
import matplotlib.pyplot as plt


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QMenu, QAction, QMessageBox
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie, QCursor
from PyQt5.QtCore import pyqtSlot, QItemSelectionModel, QModelIndex
import toolbox, collections.abc
from typing import List, Any


class MTableView(QTableView):
    def __init__(self, parent):
        super().__init__(parent)

    def contextMenuEvent(self, event):
        selection = [(i.row(), i.column()) for i in self.selectionModel().selection().indexes()]
        item_selected = self.model()._dataframe.iloc[self.selectionModel().selection().indexes()[0].row(), self.selectionModel().selection().indexes()[0].column()]
        self.menu = QMenu(self)
        
        computeAction = QAction('Compute', self)
        loadAction = QAction('Load', self)
        invalidateAction = QAction('Invalidate', self)
        expandAction = QAction('ResizeColumnToContents', self)
        plotAction = QAction('Plot*', self)
        viewRow = QAction('View row*', self)
        showtype = QAction('show type', self)
        
        if len(selection) == 1:
          def compute_subtype(i):
            if isinstance(i, RessourceHandle):
              if i.is_in_memory():
                 return compute_subtype(i.get_result())
              loader = i.get_loader()
              return (toolbox.Video if loader == toolbox.video_loader else toolbox.MatPlotLibObject if loader == toolbox.mplo_loader else RessourceHandle, [i])
            elif isinstance(i, toolbox.Video):
               return (toolbox.Video, [i])
            elif isinstance(i, toolbox.MatPlotLibObject):
               return (toolbox.Video, [i])
            elif isinstance(i, str):
               return (str, [])
            elif isinstance(i, collections.abc.Sequence):
                # print(type(i))
                rec = [compute_subtype(e)  for e in list(i)]
                # print("Rec: ", rec)
                rec = [(t, val) for t, l in rec for val in l]
                types = {t  for t, val in rec}
                if len(types) == 1:
                    return (types.pop(), [val for t, val in rec])
                else:
                   return (type(i), [])
            else:
               return (type(i), [])
                
          t, items = compute_subtype(item_selected)
          def mk_nice_type_print(s):
                return s.split('.')[-1].strip(" <>'").replace("DataFrame", "DF").replace("numpy", "np")
          def compute_subtypes_dict(i):
            if isinstance(i, RessourceHandle):
              if i.is_in_memory():
                 return {mk_nice_type_print(str(type(i))): compute_subtypes_dict(i.get_result())}
              loader = i.get_loader()
              return {mk_nice_type_print(str(type(i))): str(loader)}
            elif isinstance(i, collections.abc.Sequence):
              rec = {compute_subtypes_dict(e)  for e in list(i)}
              return {mk_nice_type_print(str(type(i))):list(rec)}
            else:
               return mk_nice_type_print(str(type(i)))
             
              
          self.menu.addAction(showtype)
          import json
          showtype.triggered.connect(lambda: self.message_box(json.dumps(compute_subtypes_dict(item_selected), separators=("\n", "\n"), indent=4)))

          if t == toolbox.Video:
              export_vid = QAction('export video', self)
              export_vid.triggered.connect(lambda: self.exportVid(items))
              self.menu.addAction(export_vid)

              view_vid = QAction('view videos', self)
              view_vid.triggered.connect(lambda: self.viewVid(items))
              self.menu.addAction(view_vid)

          if t == toolbox.MatPlotLibObject:
              view_fig = QAction('view figures', self)
              view_fig.triggered.connect(lambda: self.viewFig(items))
              self.menu.addAction(view_fig)

          if isinstance(item_selected, RessourceHandle):
            if item_selected.is_saved_on_disk():
                open_in_file_manager = QAction('open in file manager', self)
                open_in_file_manager.triggered.connect(lambda: self.open_in_file_manager(item_selected))
                self.menu.addAction(open_in_file_manager)
            if not item_selected.is_stored():
                self.menu.addAction(computeAction)
            if not item_selected.is_in_memory():
              self.menu.addAction(loadAction)
            if item_selected.is_stored():
                self.menu.addAction(invalidateAction)
                

        viewRow.triggered.connect(lambda: self.viewRowSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        computeAction.triggered.connect(lambda: self.computeSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        loadAction.triggered.connect(lambda: self.loadSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        invalidateAction.triggered.connect(lambda: self.invalidateSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        expandAction.triggered.connect(lambda: self.expandSlot({index.column() for index in self.selectionModel().selection().indexes()}))
        plotAction.triggered.connect(lambda: self.plotSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        
        if len(selection) > 1:
          self.menu.addAction(computeAction)
          self.menu.addAction(invalidateAction)
          self.menu.addAction(loadAction)
          # self.menu.addAction(showtype)

        self.menu.addAction(expandAction)
        self.menu.addAction(viewRow)
        self.menu.addAction(plotAction)
        # add other required actions
        self.menu.popup(QCursor.pos())
      
    def open_in_file_manager(self, r: RessourceHandle):
       path = r.get_disk_path()
       import subprocess
       subprocess.run(["gdbus", "call", "--session", 
                       "--dest", "org.freedesktop.FileManager1", 
                       "--object-path", "/org/freedesktop/FileManager1", 
                       "--method", "org.freedesktop.FileManager1.ShowItems", 
                       "['file://{}']".format(str(path.absolute())), ""])

    def exportVid(self, videos: List[RessourceHandle]):
       vid = [toolbox.get(video) for video in videos]
       if len(vid) == 1:
        vid = vid[0]
        path, ok = QFileDialog.getSaveFileName(self, caption="Save video to", filter="*.mp4")
        if ok:
            vid.save(path)
       elif len(vid) > 1:
          path = QFileDialog.getExistingDirectory(self, caption="Save video to")
          if not path == "":
            for i, v in enumerate(vid):
              v.save(str(pathlib.Path(path)/ f"video_{i}.mp4"))

    def viewVid(self, videos: RessourceHandle):
       for video in videos:
        vid = toolbox.get(video)
        win = self.window()
        v = toolbox.VideoPlayer()
        win.result_tabs.addTab(v, "Video")
        v.set_video(vid)
        win.menu_tabs.setCurrentWidget(win.result_tab)
        win.result_tabs.setCurrentWidget(v)

    def viewFig(self, figures: RessourceHandle):
       for figure in figures:
        win = self.window()
        fig: toolbox.MatPlotLibObject = toolbox.get(figure)
        t = fig.show(win.result_tabs)
        win.menu_tabs.setCurrentWidget(win.result_tab)
        win.result_tabs.setCurrentWidget(t)

    def message_box(self, msg):
       b = QMessageBox()
       b.setText(msg)
       b.exec()
    def computeSlot(self, selec, df):
      from toolbox.gui import Task
      win = self.window()
      task = win.mk_compute_task([(i.row(), i.column()) for i in selec], [(toolbox.RessourceHandle.is_saved_on_disk, False), (toolbox.RessourceHandle.is_saved_on_compute, True)])
    #   items = [df.iloc[i.row(), i.column()] for i in selec if isinstance(df.iloc[i.row(), i.column()], RessourceHandle)]
    #   def run(task_info):
    #       for item in task_info["progress"](items):
    #           item.get_result()
    #   task = Task(win, "compute", lambda task_info: True, lambda task_info: self.model().dataChanged.emit(selec[0], selec[-1]), run, {})
      win.add_task(task)

    def loadSlot(self, selec, df):
      from toolbox.gui import Task
      win = self.window()
      task = win.mk_compute_task([(i.row(), i.column()) for i in selec])
    #   items = [df.iloc[i.row(), i.column()] for i in selec if isinstance(df.iloc[i.row(), i.column()], RessourceHandle)]
    #   def run(task_info):
    #       for item in task_info["progress"](items):
    #           item.get_result()
    #   task = Task(win, "compute", lambda task_info: True, lambda task_info: self.model().dataChanged.emit(selec[0], selec[-1]), run, {})
      win.add_task(task)


    def plotSlot(self, selec, df):
      from toolbox.gui import Task
      win = self.window()
      task = win.mk_view_task([(i.row(), i.column()) for i in selec])
    #   items = [df.iloc[i.row(), i.column()] for i in selec if isinstance(df.iloc[i.row(), i.column()], RessourceHandle)]
    #   def run(task_info):
    #       for item in task_info["progress"](items):
    #           item.get_result()
    #   task = Task(win, "compute", lambda task_info: True, lambda task_info: self.model().dataChanged.emit(selec[0], selec[-1]), run, {})
      win.add_task(task)

    def viewRowSlot(self, selec, df):
      from toolbox.gui import Task
      win = self.window()
      for i in {s.row() for s in selec}:
        win.view_row(df.iloc[i,:])

    def invalidateSlot(self, selec, df):
      from toolbox.gui import Task
      win = self.window()
      items: List[toolbox.RessourceHandle] = [df.iloc[i.row(), i.column()] for i in selec if isinstance(df.iloc[i.row(), i.column()], RessourceHandle)]
      def run(task_info):
          for item in task_info["progress"](items):
              item.invalidate_all()
      task = Task(win, "invalidate", lambda task_info: True, lambda task_info: self.model().dataChanged.emit(selec[0], selec[-1]), run, {})
      win.add_task(task)

    def expandSlot(self, indices):
      for i in indices:
        self.resizeColumnToContents(i)
    #   self.parent.add_task()
    #   for ind in selec:
    #       item = df.iloc[ind.row(), ind.column()]
    #       if is
    #   item.get_result()
    #   self.model().dataChanged.emit(updatel[0])

