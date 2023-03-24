from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import re

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

  rnum = re.compile("(\d+)/{}".format(len(max_len)) if max_len else "#(\d+)")
  text = "{}/{}".format(i+1, max_len) if max_len else "#{}".format(i+1)
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


    while not mexit:
      show()