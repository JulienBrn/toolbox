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
                col = str(self._dataframe.columns[section])
                return col if not col[0] == "_" else col[1:]
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
            import numpy as np
            if hasattr(val, "__len__") and not isinstance(val, str) and not isinstance(val, np.ndarray) and len(str(val)) > 100:
                types = {str(type(t)).split(".")[-1].replace("class", "").strip("\' <>,") for t in val}
                valtypestr = str(type(val)).replace('class', '').strip('\' <>,')
                if len(types)==1:
                    return f"{valtypestr}({len(val)}, {types.pop()})"
                if len(types) <3:
                    return f"{valtypestr}({len(val)}, {types})"
                else:
                    return f"{valtypestr}({len(val)})"
            else:
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
    

from toolbox.victor_stuff import Video
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableView, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QIcon, QImage, QStandardItem, QStandardItemModel, QMovie, QCursor
from PyQt5.QtCore import pyqtSlot, QItemSelectionModel, QModelIndex
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore

class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        # self.setSizePolicy(sizePolicy)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget_6 = self
        self.m_view = GraphicsView(self.widget_6)
        self.m_scene = QtWidgets.QGraphicsScene(self.widget_6)
        self.m_view.setScene(self.m_scene)
        self.construct_player()
        self.construct_scene()
        self.timer = QtCore.QTimer()
        self.vid = None
        self.pushButton.released.connect(lambda: self.pause() if self.timer.isActive() else self.play())
        
        self.timer.timeout.connect(lambda: self.next_frame())

    def construct_player(self):
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget_6)
        self.verticalLayout_10.setSpacing(3)
        self.verticalLayout_10.setContentsMargins(100, 100, 100, 100)
        self.verticalLayout_10.addWidget(self.m_view)
        self.horizontalSlider = QtWidgets.QSlider(self.widget_6)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(0, 0))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.verticalLayout_10.addWidget(self.horizontalSlider)
        self.widget_7 = QtWidgets.QWidget(self.widget_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_5.setContentsMargins(-1, 0, -1, 0)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.label_9 = QtWidgets.QLabel(self.widget_7)
        self.horizontalLayout_5.addWidget(self.label_9)
        self.spinBox = QtWidgets.QSpinBox(self.widget_7)
        self.horizontalLayout_5.addWidget(self.spinBox)
        self.label_10 = QtWidgets.QLabel(self.widget_7)
        self.horizontalLayout_5.addWidget(self.label_10)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.widget_7)
        self.horizontalLayout_5.addWidget(self.doubleSpinBox)
        self.label_11 = QtWidgets.QLabel(self.widget_7)
        self.horizontalLayout_5.addWidget(self.label_11)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.widget_7)
        self.doubleSpinBox_2.setSingleStep(0.05)
        self.doubleSpinBox_2.setProperty("value", 1.0)
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.pushButton = QtWidgets.QPushButton(self.widget_7)
        self.pushButton.setMinimumSize(QtCore.QSize(30, 0))
        self.pushButton.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushButton.setIconSize(QtCore.QSize(0, 0))
        self.pushButton.setCheckable(False)
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.verticalLayout_10.addWidget(self.widget_7)

        self.label_9.setText("Frame")
        self.label_10.setText("Time")
        self.label_11.setText("Speed")
        self.pushButton.setText("P")
        

    def construct_scene(self):
        self.qpixmapitem = QtWidgets.QGraphicsPixmapItem()
        self.m_scene.addItem(self.qpixmapitem)

    def set_image(self, img):
        import cv2
        # logger.warning(img.shape)
        cvRGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # logger.warning(cvRGBImg.shape)
        self.qimg = QtGui.QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], 3*cvRGBImg.shape[1], QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(self.qimg)
        self.qpixmapitem.setPixmap(self.pixmap)
        
        # self.m_view.fitInView(self.m_scene.sceneRect())
        self.m_scene.update()
        # self.m_view.setSceneRect(-100, -100, 2000, 2000)
        # print(self.qpixmapitem.boundingRect())
        # print(self.m_scene.sceneRect())
        # print(self.m_view.setFixedSize(500, 500))
        self.m_view.fitInView(0,0, cvRGBImg.shape[1], cvRGBImg.shape[0], QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        # self.m_view.fitInView(-100, -100, 2000, 2000)
        self.m_view.update()
        self.m_scene.update()
        self.m_view.update()
        # self.m_view.fitInView(0,0, cvRGBImg.shape[0], cvRGBImg.shape[1])

    def set_video(self, vid):
        self.pause()
        self.vid = vid.copy()
        self.set_image(vid[0])
        self.vid.__iter__()
        self.horizontalSlider.setMaximum(self.vid.nb_frames)
        self.spinBox.setMaximum(self.vid.nb_frames)
        self.doubleSpinBox.setMaximum(int(self.vid.nb_frames)*self.vid.fps)
        # self.timer.timeout.connect(next_frame)
        # self.timer.start(int(1000/vid.fps))
    def next_frame(self):
        try:
            self.set_image(self.vid.__next__())
        except StopIteration:
            self.vid.__iter__()
            self.set_image(self.vid.__next__())
        self.horizontalSlider.setValue(self.vid.iterpos)
        self.spinBox.setValue(self.vid.iterpos)
        self.doubleSpinBox.setValue(self.vid.iterpos/self.vid.fps)
    def play(self):
        self.timer.start(int(1000/self.vid.fps))
        self.spinBox.setDisabled(True)
        self.doubleSpinBox.setDisabled(True)
        self.pushButton.setText("Pause")

    def pause(self):
        self.timer.stop()
        self.spinBox.setDisabled(False)
        self.doubleSpinBox.setDisabled(False)
        self.pushButton.setText("Play")