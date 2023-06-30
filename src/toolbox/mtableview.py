from __future__ import annotations
from typing import List, Tuple, Dict, Any
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


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QMenu, QAction, QMessageBox
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie, QCursor
from PyQt5.QtCore import pyqtSlot, QItemSelectionModel, QModelIndex
import toolbox
from typing import List


class MTableView(QTableView):
    def __init__(self, parent):
        super().__init__(parent)

    def contextMenuEvent(self, event):
        selection = [(i.row(), i.column()) for i in self.selectionModel().selection().indexes()]
        self.menu = QMenu(self)
        computeAction = QAction('Compute', self)
        loadAction = QAction('Load', self)
        invalidateAction = QAction('Invalidate', self)
        expandAction = QAction('ResizeColumnToContents', self)
        plotAction = QAction('Plot', self)
        viewRow = QAction('View row', self)
        showtype = QAction('show type', self)
        viewRow.triggered.connect(lambda: self.viewRowSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        computeAction.triggered.connect(lambda: self.computeSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        loadAction.triggered.connect(lambda: self.loadSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        invalidateAction.triggered.connect(lambda: self.invalidateSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        expandAction.triggered.connect(lambda: self.expandSlot({index.column() for index in self.selectionModel().selection().indexes()}))
        plotAction.triggered.connect(lambda: self.plotSlot(self.selectionModel().selection().indexes(), self.model()._dataframe))
        showtype.triggered.connect(lambda: self.message_box(str(type(self.model()._dataframe.iloc[self.selectionModel().selection().indexes()[0].row(), self.selectionModel().selection().indexes()[0].column()]))))
        self.menu.addAction(computeAction)
        self.menu.addAction(invalidateAction)
        self.menu.addAction(expandAction)
        self.menu.addAction(plotAction)
        self.menu.addAction(viewRow)
        self.menu.addAction(loadAction)
        self.menu.addAction(showtype)
        # add other required actions
        self.menu.popup(QCursor.pos())
      
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

