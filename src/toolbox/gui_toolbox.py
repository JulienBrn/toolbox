import typing
from PyQt5 import QtCore
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import  QModelIndex
from PyQt5.QtGui import QIcon, QImage, QPixmap
import PyQt5.QtGui as QtGui
import pandas as pd
from toolbox import RessourceHandle
import logging

logger = logging.getLogger(__name__)

class DisplayImg:
    def __init__(self, img):
        self.image = QPixmap.fromImage(img)

class DataFrameModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df.copy()

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._dataframe.columns[section])
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = index.row()
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype
        if row >= len(self._dataframe.index):
            logger.error("Invalid index {} to acces dataframe of size {}".format(row, len(self._dataframe.index)))
            return QtCore.QVariant()
        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            # if isinstance(val, DisplayImg):
            #     print("Showing img")
            #     return val.image
            # else:
                return str(val)
        # elif role == DataFrameModel.DecorationRole:
        #     print("decoration")
        elif role == DataFrameModel.ValueRole:
            return val
        elif role == DataFrameModel.DtypeRole:
            return dt
        elif role==QtCore.Qt.BackgroundRole:
            if "Discarded" in self._dataframe.columns and self._dataframe["Discarded"].iat[row] == True:
                return QtGui.QColor('red')

        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles
    def sort(self, col, asc):
        def mkey(col):
            def custom_key(x):
                param1 = str(type(x))
                if isinstance(x, RessourceHandle):
                    if x.is_in_memory():
                        param2 = 10
                        if hasattr(x.get_result(), "shape"):
                            param3= x.get_result().shape
                        elif hasattr(x.get_result(), "__len__"):
                            param3 = (len(x),)
                        else:
                            param3 = 0
                        param4 = str(x.get_result())
                    else:
                        param2 = 2 if x.is_saved_on_disk() else 1
                        param3=(0,)
                        param4=str(x)
                else:
                    param2 = 0
                    param3=(0,)
                    param4=str(x)
                return (param1, param2, param3, param4)
            return col.apply(lambda x: custom_key(x))
        try:
            self._dataframe.sort_values(by=self._dataframe.columns[col], inplace=True, ascending=asc)
            self._dataframe.reset_index(drop=True, inplace=True)
        except:
            self._dataframe.sort_values(by=self._dataframe.columns[col], inplace=True, ascending=asc, key=mkey)
            self._dataframe.reset_index(drop=True, inplace=True)
        self.dataChanged.emit(
            self.createIndex(0,0), self.createIndex(len(self._dataframe.index), len(self._dataframe.columns)), (QtCore.Qt.EditRole,)
        ) 
    




from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QTreeView
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie
from PyQt5.QtCore import QAbstractItemModel, QModelIndex


class DictTreeModel(QStandardItemModel):
    def __init__(self, df: pd.DataFrame=pd.DataFrame(), sep="."):
        super().__init__()
        self.set_values(df, sep)

    def set_values(self, df, sep="."):
        self.item_dict=[self.invisibleRootItem(), [None for i in range(1, len(df.columns))], {}]
        self.setHorizontalHeaderLabels(df.columns)
        self.cols = df.columns
        for i in range(len(df.index)):
            keys = df.iat[i, 0].split(".")
            curr = self.item_dict
            for k in keys:
                if not k in curr[2]:
                    curr[0].appendRow([QStandardItem(k)] +[QStandardItem("") for i in range(len(df.columns)-1)])
                    for j in range(len(df.columns)-1):
                        if not curr[1][j] is None:
                            curr[1][j].setEditable(False)
                    for j in range(1, len(df.columns)):
                        curr[0].child(curr[0].rowCount() - 1, j).setEditable(True)
                    curr[0].child(curr[0].rowCount() - 1, 0).setEditable(False)
                    curr[2][k]=[curr[0].child(curr[0].rowCount() - 1, 0), [curr[0].child(curr[0].rowCount() - 1, j) for j in range(1, len(df.columns))], {}]
                curr = curr[2][k]
            for j in range(len(df.columns)-1):
                curr[1][j].setText(str(df.iloc[i, j+1]))


    def get_values(self, sep="."):
        if len(self.cols) >0:
            params = {}
            def rec_print(root, prefix):
                if root[2] == {}:
                    params[prefix+root[0].text()] = [root[1][j].text() for j in range(len(self.cols)-1)]
                for child, val in root[2].items():
                    rec_print(val, prefix+root[0].text()+".")
            for child, val in self.item_dict[2].items():
                rec_print(val, "")
            return pd.DataFrame.from_dict(params, orient="index", columns=self.cols[1:]).reset_index(names=self.cols[0])
        else:
            return pd.DataFrame()
        
    # def headerData(self, section: int, orientation, role: int) :
    #     if role != QtCore.Qt.DisplayRole: return
    #     return str("t1\nt2")



















class DictTreeModelFull(QAbstractItemModel):
    def __init__(self, df: pd.DataFrame, key_col = None, sep="."):
        super().__init__()

        self.df=df
        self.rootindex = self.createIndex(0, 0, 1)
        self.nbcols=len(df.columns)
        self.maxrows=5
        self.sep = sep
        print(self.rootindex)
        print("rootindex id = {}".format(self.rootindex.internalId()))
        exit()
    def get_list_from_index(self, index):
        pass
    def create_dict(self, l):
        d = {}
        for s in l:
            tokens = s.split(self.sep)




    def index(self, row, col, parentindex):
        if not parentindex.isValid():
            print("Returning rootindex as index")
            return self.rootindex
        print("Creating index {}".format("({}, {}, {})".format(row, col, parentindex.internalId())))
        return self.createIndex(row, col, parentindex.internalId()*(self.nbcols*self.maxrows)+row*self.nbcols+col)
    def parent(self, index):
        if index != self.rootindex:
            parentid = int((index.internalId() - index.column()-index.row()*self.nbcols)/(self.nbcols*self.maxrows))
            return self.createIndex(int(parentid / self.nbcols) % self.maxrows, 0, parentid)
        else:
            print("Invalid Parent for index {}".format("({}, {}, {})".format(index.row(), index.column(), index.internalId())))
            # raise BaseException("stop")
            return QModelIndex()
    def rowCount(self, index):
        if index == self.rootindex:
            print("returning 3 for index {}".format("({}, {}, {})".format(index.row(), index.column(), index.internalId())))
            return 3
        elif not index.isValid():
            return 1
        else:
            return self.rowCount(index.parent()) 
            print("returning {}".format(max(0, self.rowCount(index.parent()) -1)))
            return max(0, self.rowCount(index.parent()) -1)
    def columnCount(self, index):
        return self.nbcols
        if not index.parent().isValid():
            return 3
        else:
            return max(0, self.columnCount(index.parent()) -1)
        # return len(self.df.columns)
    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            # if isinstance(val, DisplayImg):
            #     print("Showing img")
            #     return val.image
            # else:
                return str("data")
        # elif role == DataFrameModel.DecorationRole:
        #     print("decoration")
        if role == DataFrameModel.ValueRole:
            return "data"
        if role == DataFrameModel.DtypeRole:
            return str
        return QtCore.QVariant()
        return "data"