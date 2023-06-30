
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QTreeView
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie, QAbstractItemModel, QModelIndex

import pandas as pd


class DictTreeModelIndex(QModelIndex):
    def __init__(self, str, row, column, parent):
        super().__init__(self, row, column, parent)
        self.key = str

class DictTreeModel(QAbstractItemModel):
    def __init__(self, df: pd.DataFrame, key_col=None, sep="."):
        super.__init__(self)
        self.df=df
        self.key_col = key_col
        self.sep=sep
    def index(self, row, col, parent):
        return self.createIndex(row, col, parent)
    def parent(self, index):
        return index.parent()
    def row_count(self, index):
        return 3
    def column_count(self, index):
        return len(self.df.columns)
    def data(self, index):
        return "data"
    

