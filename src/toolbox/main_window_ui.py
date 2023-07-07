# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/toolbox/ui/main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(971, 709)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.menu_tabs = QtWidgets.QTabWidget(self.centralwidget)
        self.menu_tabs.setObjectName("menu_tabs")
        self.setup_tab = QtWidgets.QWidget()
        self.setup_tab.setObjectName("setup_tab")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.setup_tab)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.setup_params_tree = QtWidgets.QTreeView(self.setup_tab)
        self.setup_params_tree.setObjectName("setup_params_tree")
        self.verticalLayout_5.addWidget(self.setup_params_tree)
        self.setup_button_menu = QtWidgets.QWidget(self.setup_tab)
        self.setup_button_menu.setObjectName("setup_button_menu")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.setup_button_menu)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.export_params = QtWidgets.QPushButton(self.setup_button_menu)
        self.export_params.setObjectName("export_params")
        self.horizontalLayout_3.addWidget(self.export_params)
        self.load_params = QtWidgets.QPushButton(self.setup_button_menu)
        self.load_params.setObjectName("load_params")
        self.horizontalLayout_3.addWidget(self.load_params)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout_5.addWidget(self.setup_button_menu)
        self.menu_tabs.addTab(self.setup_tab, "")
        self.computation_tab = QtWidgets.QWidget()
        self.computation_tab.setObjectName("computation_tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.computation_tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.splitter = QtWidgets.QSplitter(self.computation_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.splitter.setLineWidth(2)
        self.splitter.setMidLineWidth(2)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(8)
        self.splitter.setObjectName("splitter")
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.dataframe_list = QtWidgets.QTreeView(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dataframe_list.sizePolicy().hasHeightForWidth())
        self.dataframe_list.setSizePolicy(sizePolicy)
        self.dataframe_list.setMinimumSize(QtCore.QSize(10, 0))
        self.dataframe_list.setObjectName("dataframe_list")
        self.verticalLayout.addWidget(self.dataframe_list)
        self.view_params = QtWidgets.QTreeView(self.widget)
        self.view_params.setObjectName("view_params")
        self.verticalLayout.addWidget(self.view_params)
        self.widget1 = QtWidgets.QWidget(self.splitter)
        self.widget1.setObjectName("widget1")
        self.verticalLayout1 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout1.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout1.setObjectName("verticalLayout1")
        self.search_filter_menu = QtWidgets.QWidget(self.widget1)
        self.search_filter_menu.setObjectName("search_filter_menu")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.search_filter_menu)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.query = QtWidgets.QLineEdit(self.search_filter_menu)
        self.query.setObjectName("query")
        self.horizontalLayout.addWidget(self.query)
        self.previous = QtWidgets.QPushButton(self.search_filter_menu)
        self.previous.setObjectName("previous")
        self.horizontalLayout.addWidget(self.previous)
        self.next = QtWidgets.QPushButton(self.search_filter_menu)
        self.next.setObjectName("next")
        self.horizontalLayout.addWidget(self.next)
        self.verticalLayout1.addWidget(self.search_filter_menu)
        self.tableView = MTableView(self.widget1)
        self.tableView.setObjectName("tableView")
        self.tableView.horizontalHeader().setCascadingSectionResizes(False)
        self.tableView.horizontalHeader().setSortIndicatorShown(True)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.verticalLayout1.addWidget(self.tableView)
        self.compute_btn_menu = QtWidgets.QWidget(self.widget1)
        self.compute_btn_menu.setObjectName("compute_btn_menu")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.compute_btn_menu)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.export_btn = QtWidgets.QPushButton(self.compute_btn_menu)
        self.export_btn.setObjectName("export_btn")
        self.horizontalLayout_2.addWidget(self.export_btn)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.compute = QtWidgets.QPushButton(self.compute_btn_menu)
        self.compute.setObjectName("compute")
        self.horizontalLayout_2.addWidget(self.compute)
        self.view = QtWidgets.QPushButton(self.compute_btn_menu)
        self.view.setObjectName("view")
        self.horizontalLayout_2.addWidget(self.view)
        self.verticalLayout1.addWidget(self.compute_btn_menu)
        self.verticalLayout_3.addWidget(self.splitter)
        self.menu_tabs.addTab(self.computation_tab, "")
        self.result_tab = QtWidgets.QWidget()
        self.result_tab.setObjectName("result_tab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.result_tab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.result_tabs = QtWidgets.QTabWidget(self.result_tab)
        self.result_tabs.setElideMode(QtCore.Qt.ElideRight)
        self.result_tabs.setTabsClosable(True)
        self.result_tabs.setMovable(True)
        self.result_tabs.setObjectName("result_tabs")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_3 = QtWidgets.QWidget(self.tab)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_7.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.widget_3)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_7.addWidget(self.label_4)
        self.gridLayout.addWidget(self.widget_3, 0, 2, 1, 2)
        self.widget_2 = QtWidgets.QWidget(self.tab)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label = QtWidgets.QLabel(self.widget_2)
        self.label.setObjectName("label")
        self.verticalLayout_6.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.gridLayout.addWidget(self.widget_2, 0, 0, 1, 2)
        self.widget_4 = QtWidgets.QWidget(self.tab)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_7 = QtWidgets.QLabel(self.widget_4)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_9.addWidget(self.label_7)
        self.widget_6 = QtWidgets.QWidget(self.widget_4)
        self.widget_6.setObjectName("widget_6")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget_6)
        self.verticalLayout_10.setSpacing(3)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.graphicsView = QtWidgets.QGraphicsView(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_10.addWidget(self.graphicsView)
        self.horizontalSlider = QtWidgets.QSlider(self.widget_6)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(0, 0))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.verticalLayout_10.addWidget(self.horizontalSlider)
        self.widget_7 = QtWidgets.QWidget(self.widget_6)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_5.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.label_9 = QtWidgets.QLabel(self.widget_7)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_5.addWidget(self.label_9)
        self.spinBox = QtWidgets.QSpinBox(self.widget_7)
        self.spinBox.setMaximum(10000)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_5.addWidget(self.spinBox)
        self.label_10 = QtWidgets.QLabel(self.widget_7)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_5.addWidget(self.label_10)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.widget_7)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox)
        self.label_11 = QtWidgets.QLabel(self.widget_7)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_5.addWidget(self.label_11)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.widget_7)
        self.doubleSpinBox_2.setSingleStep(0.05)
        self.doubleSpinBox_2.setProperty("value", 1.0)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.pushButton = QtWidgets.QPushButton(self.widget_7)
        self.pushButton.setMinimumSize(QtCore.QSize(30, 0))
        self.pushButton.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushButton.setIconSize(QtCore.QSize(0, 0))
        self.pushButton.setCheckable(False)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.verticalLayout_10.addWidget(self.widget_7)
        self.verticalLayout_9.addWidget(self.widget_6)
        self.label_8 = QtWidgets.QLabel(self.widget_4)
        self.label_8.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_8.setObjectName("label_8")
        self.verticalLayout_9.addWidget(self.label_8)
        self.gridLayout.addWidget(self.widget_4, 1, 0, 1, 2)
        self.widget_5 = QtWidgets.QWidget(self.tab)
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_6 = QtWidgets.QLabel(self.widget_5)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_8.addWidget(self.label_6)
        self.label_5 = QtWidgets.QLabel(self.widget_5)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_8.addWidget(self.label_5)
        self.gridLayout.addWidget(self.widget_5, 1, 2, 1, 2)
        self.result_tabs.addTab(self.tab, "")
        self.verticalLayout_4.addWidget(self.result_tabs)
        self.exportall = QtWidgets.QPushButton(self.result_tab)
        self.exportall.setObjectName("exportall")
        self.verticalLayout_4.addWidget(self.exportall)
        self.menu_tabs.addTab(self.result_tab, "")
        self.verticalLayout_2.addWidget(self.menu_tabs)
        self.status_bar = QtWidgets.QWidget(self.centralwidget)
        self.status_bar.setMinimumSize(QtCore.QSize(0, 28))
        self.status_bar.setObjectName("status_bar")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.status_bar)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.current_exec = QtWidgets.QLabel(self.status_bar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_exec.sizePolicy().hasHeightForWidth())
        self.current_exec.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(6)
        self.current_exec.setFont(font)
        self.current_exec.setObjectName("current_exec")
        self.horizontalLayout_4.addWidget(self.current_exec)
        self.progressBar = QtWidgets.QProgressBar(self.status_bar)
        font = QtGui.QFont()
        font.setPointSize(6)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_4.addWidget(self.progressBar)
        self.aborttask = QtWidgets.QPushButton(self.status_bar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.aborttask.sizePolicy().hasHeightForWidth())
        self.aborttask.setSizePolicy(sizePolicy)
        self.aborttask.setMinimumSize(QtCore.QSize(20, 0))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.aborttask.setFont(font)
        self.aborttask.setObjectName("aborttask")
        self.horizontalLayout_4.addWidget(self.aborttask)
        self.verticalLayout_2.addWidget(self.status_bar)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 971, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.menu_tabs.setCurrentIndex(2)
        self.result_tabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.export_params.setText(_translate("MainWindow", "Export Params"))
        self.load_params.setText(_translate("MainWindow", "Load Params"))
        self.menu_tabs.setTabText(self.menu_tabs.indexOf(self.setup_tab), _translate("MainWindow", "Setup"))
        self.previous.setText(_translate("MainWindow", "<"))
        self.next.setText(_translate("MainWindow", ">"))
        self.export_btn.setText(_translate("MainWindow", "Export"))
        self.compute.setText(_translate("MainWindow", "Compute All"))
        self.view.setText(_translate("MainWindow", "View All"))
        self.menu_tabs.setTabText(self.menu_tabs.indexOf(self.computation_tab), _translate("MainWindow", "Computations"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_9.setText(_translate("MainWindow", "Frame"))
        self.label_10.setText(_translate("MainWindow", "Time"))
        self.label_11.setText(_translate("MainWindow", "Speed"))
        self.pushButton.setText(_translate("MainWindow", "P"))
        self.label_8.setText(_translate("MainWindow", "TextLabel"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))
        self.result_tabs.setTabText(self.result_tabs.indexOf(self.tab), _translate("MainWindow", "Page"))
        self.exportall.setText(_translate("MainWindow", "Export All Figures"))
        self.menu_tabs.setTabText(self.menu_tabs.indexOf(self.result_tab), _translate("MainWindow", "Results"))
        self.current_exec.setText(_translate("MainWindow", "All Done"))
        self.aborttask.setText(_translate("MainWindow", "X"))
from toolbox.mtableview import MTableView
