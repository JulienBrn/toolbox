import toolbox
import numpy as np

mat = np.array([
    [np.nan, 1], 
    [2, 2],
    [6, np.nan],
    [5, 5]
])

res = toolbox.order_differences(mat, 0)
print("Final result :", res)