import PyQt5.QtCore as QtCore
from PyQt5.QtCore import  QModelIndex
from PyQt5.QtGui import QIcon, QImage, QPixmap
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
                return self._dataframe.columns[section]
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
        if role == DataFrameModel.DtypeRole:
            return dt
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

