from PyQt5 import QtGui
import  PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
  def __init__(self):
    self.fig = Figure()
    self.ax = None
    FigureCanvas.__init__(self, self.fig)
    FigureCanvas.updateGeometry(self)

class MplWidget(PyQt5.QtWidgets.QWidget):
  def __init__(self, parent = None):
    PyQt5.QtWidgets.QWidget.__init__(self, parent)
    self.canvas = MplCanvas()
    self.vbl = PyQt5.QtWidgets.QVBoxLayout()
    self.vbl.addWidget(self.canvas)
    self.setLayout(self.vbl)
  def reset(self, parent = None):
    self.canvas = MplCanvas()
    for i in reversed(range(self.vbl.count())): 
      widgetToRemove = self.vbl.itemAt(i).widget()
      self.vbl.removeWidget(widgetToRemove)
      widgetToRemove.setParent(None)
    self.vbl.addWidget(self.canvas)
    # self.setLayout(self.vbl)