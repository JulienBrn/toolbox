import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def order_differences(mat, start = None):
    res=[]
    if mat.shape[0]==0:
        return res
    if start is None:
       start = random.randint(0, mat.shape[0]) 
  
    mask=np.full((mat.shape[0]), False)
    distances = euclidean_distances(mat,mat)
    
    mask[start] = True
    res.append(start)

    while len(res) < mat.shape[0]:
      group_distances= np.min(distances[:,mask], axis=1)
      index = np.ma.masked_array(group_distances,mask).argmax()
      mask[index]=True
      res.append(index)
    
    return np.argsort(res)


    