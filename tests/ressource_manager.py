from toolbox import Manager, np_loader, df_loader, float_loader
import beautifullogger

beautifullogger.setup()

m = Manager("./tests/cache")
r = m.declare_ressource("tests/content.tsv", df_loader, "test_ressource")

def multiply(df, i):
    print("multiply called with df=\n{}\ni= {}\nColumns:{}".format(df, i, df.columns))
    copy = df.copy()
    copy["x"] *= i
    return copy

mult = m.declare_computable_ressource(multiply, {"df":r, "i":2}, df_loader, "mult_block")

def make_avg_dict(df):
    return {"x" : df["x"].mean(), "y" : df["y"].mean()}

start_avgs = m.declare_computable_ressources(make_avg_dict, {"df":r}, 
  {
    "x": (float_loader, "x_avg", True),
    "y": (float_loader, "y_avg", True),
  })

end_avgs = m.declare_computable_ressources(make_avg_dict, {"df":mult}, 
  {
    "x": (float_loader, "x_avg_e", True),
    "y": (float_loader, "y_avg_e", False),
  })

print(mult.get_result())
mult.save()

end_result_avgs = {key:val.get_result() for key, val in end_avgs.items()}
start_result_avgs = {key:val.get_result() for key, val in start_avgs.items()}


print(start_result_avgs)
print(end_result_avgs)