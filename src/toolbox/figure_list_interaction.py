from typing import Callable, Any
import matplotlib, matplotlib.pyplot as plt
import re, logging

logger = logging.getLogger(__name__)


class FigureList:
  i: int
  max_len: int | None
  exit: bool
  rnum : re
  text = None
  textbox = None
  has_fig = False
  load_data_func: Callable[[int], Any]
  draw_fig: Callable[[Any], plt.Figure]
  
  loaded_data={}
  
  def __init__(self, 
    load_data_func: Callable[[int], Any], 
    draw_fig: Callable[[Any], plt.Figure],
    max_len = None,
    start_figure=0
  ):
    self.max_len=max_len
    self.i=start_figure
    self.load_data_func = load_data_func
    self.draw_fig = draw_fig
    self.rnum = re.compile("(\d+)/{}".format(self.max_len) if self.max_len else "#(\d+)")
    self.exit=False

  def show_fig(self, gen, first= False):
    for ind in gen:
      if ind in self.loaded_data:
        data = self.loaded_data[ind]
      else:
          data = self.load_data_func(ind)
          if data is None:
            continue
          self.loaded_data[ind] = data
      self.i = ind
      if not first:
        plt.close()
      self.fig = self.draw_fig(data)
      self.add_interface(self.fig)
      self.has_fig=True
      return True
    logger.error("No figure found")
    return False

  def show(self):
     self.next_figure(include_cur=True)
     while not self.exit:
        logger.info("showing")
        if not self.has_fig:
            logger.error("No figure to be shown... Strange...")
            break
        self.has_fig=False
        plt.show()


  def add_interface(self, fig):
    self.ax_exitbutton = plt.axes([0.90, 0.97, 0.10, 0.03])
    self.exitbtn = matplotlib.widgets.Button(self.ax_exitbutton, 'Exit all', color='white', hovercolor='grey')
    self.exitbtn.on_clicked(lambda _: self.exit_all())
    
    self.ax_numtext = plt.axes([0.045, 0.97, 0.06, 0.03])
    text = "{}/{}".format(self.i+1, self.max_len) if self.max_len else "#{}".format(self.i+1)
    self.textbox = matplotlib.widgets.TextBox(self.ax_numtext, "", initial=text)
    self.textbox.on_submit(lambda t: self.go_to_figure(t))

    self.ax_nextbutton = plt.axes([0.11, 0.97, 0.04, 0.03])
    self.nextbtn = matplotlib.widgets.Button(self.ax_nextbutton, '=>', color='white', hovercolor='grey')
    self.nextbtn.on_clicked(lambda _: self.next_figure())

    self.ax_prevbutton = plt.axes([0.00, 0.97, 0.04, 0.03])
    self.prevbtn = matplotlib.widgets.Button(self.ax_prevbutton, '<=', color='white', hovercolor='grey')
    self.prevbtn.on_clicked(lambda _: self.prev_figure())

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

  def exit_all(self):
    self.exit = True
    logger.info("Closing")
    plt.close()

  def reload(self, erase_data=True):
    if erase_data:
      del self.loaded_data[self.i]
    self.next_figure(include_cur=True)

  def next_figure(self, include_cur =False):
    def generator():
      mi = self.i % self.max_len if self.max_len else self.i
      if include_cur:
        yield mi
      while True:
        mi= (mi + 1) % self.max_len if self.max_len else (mi+1)
        yield mi
    self.show_fig(generator())

  def prev_figure(self):
    def generator():
      mi = self.i
      while True:
        mi= (mi - 1) % self.max_len if self.max_len else (mi-1)
        yield mi
    self.show_fig(generator())

  def go_to_figure(self, t):
    matching = self.__rnum.fullmatch(t)
    if matching:
      inext = int(str(matching.group(1)))
      self.show_fig(inext-1)
    else:
      self.textbox.set_val("{}/{}".format(self.i+1, self.max_len) if self.max_len else "#{}".format(self.i+1))
  


def make_figure_list_interaction(
  make_fig_func: Callable[[int], plt.Figure], 
  max_len = None,
  index_ok = lambda i: True,
  start_figure=0
):
  
  mexit=False
  i = start_figure
  
  get_new_val = (lambda val: val % max_len) if max_len else (lambda val: val)

  def next_figure(_):
    nonlocal i 
    i=get_new_val(i+1)
    while not index_ok(i):
      i=get_new_val(i+1)
    plt.close()

  def prev_figure(_): 
    nonlocal i 
    i=get_new_val(i-1)
    while not index_ok(i):
       i=get_new_val(i-1)
    plt.close()

  def call_exit(_):
    nonlocal mexit
    plt.close()
    mexit= True

  rnum = re.compile("(\d+)/{}".format(max_len) if max_len else "#(\d+)")
  text = None
  textbox = None

  def go_to_figure(t):
    nonlocal i, rnum, text, textbox
    matching = rnum.fullmatch(t)
    if matching:
      inext = int(str(matching.group(1)))
      if index_ok(get_new_val(inext-1)):
        i=get_new_val(inext-1)
        plt.close()
        return 
    textbox.set_val(text)
    


  def show():
    nonlocal i, text, textbox
    fig = make_fig_func(i)

    ax_exitbutton = plt.axes([0.90, 0.97, 0.10, 0.03])
    exitbtn = matplotlib.widgets.Button(ax_exitbutton, 'Exit all', color='white', hovercolor='grey')
    exitbtn.on_clicked(call_exit)
    
    ax_numtext = plt.axes([0.045, 0.97, 0.06, 0.03])
    text = "{}/{}".format(i+1, max_len) if max_len else "#{}".format(i+1)
    textbox = matplotlib.widgets.TextBox(ax_numtext, "", initial=text)
    textbox.on_submit(go_to_figure)

    ax_nextbutton = plt.axes([0.11, 0.97, 0.04, 0.03])
    nextbtn = matplotlib.widgets.Button(ax_nextbutton, '=>', color='white', hovercolor='grey')
    nextbtn.on_clicked(next_figure)

    ax_prevbutton = plt.axes([0.00, 0.97, 0.04, 0.03])
    prevbtn = matplotlib.widgets.Button(ax_prevbutton, '<=', color='white', hovercolor='grey')
    prevbtn.on_clicked(prev_figure)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

  while not index_ok(i):
    i = get_new_val(i+1)
  while not mexit:
    show()