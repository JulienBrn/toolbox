import toolbox
import logging, beautifullogger
import numpy as np
beautifullogger.setup()

arr= np.zeros(300)
arr[1]=2000
arr[2]=2000

arr[5]=2000
arr[6]=2000
arr[7]=2000

arr[9] = 2000
arr[10] = 2000
arr[11] = 2000

arr[13] = 2000
arr[14] = 2000

arr[16] = 40000
arr[17] = 40000
arr[18] = 40000

arr[26] = 500
arr[27] = 500
arr[28] = 500

arr[-1] = 2000
arr[-2] = 2000
arr[-3] = 2000
print(arr.shape)
res = toolbox.replace_artefacts_with_nans2(arr, 1, deviation_factor=1, min_length=3, join_width=3, recursive=True, shoulder_width=0)

print("res=\n", res[2])