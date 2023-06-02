import random
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
import itertools
import collections

def order_differences(mat, start=None):
    res = []
    if mat.shape[0] == 0:
        return res
    if start is None:
        start = random.randint(0, mat.shape[0])

    mask = np.full((mat.shape[0]), False)
    distances = nan_euclidean_distances(mat, mat)

    mask[start] = True
    res.append(start)

    while len(res) < mat.shape[0]:
        group_distances = np.min(distances[:, mask], axis=1)
        index = np.ma.masked_array(group_distances, mask).argmax()
        mask[index] = True
        res.append(index)

    return np.argsort(res)

class crange(): #should inherit from collections.abc.Collection
    def __init__(self, center, max_left, max_right=None , step=1, direction_right=None):
        if max_right is None:
            max_right= max_left
            max_left = center - (max_right - center)
        if direction_right is None:
            self.direction_right = (max_right - center) >= (center - max_left)


        self.center = center
        self.step = step
        self.mleft = int((max_left - center)/step)
        self.mright= int((max_right - center)/step)
        # print(self.mleft, self.mright)

    def __len__(self): return self.mright - self.mleft +1

    def __iter__(self): 
        yield(self.center + self.step * 0)
        for i in range(1, max(self.mright, - self.mleft)+1):
            if self.direction_right:
                if i <= self.mright:
                    yield(self.center + self.step * i)
                if -i >= self.mleft:
                    yield(self.center + self.step * (-i))
            else:
                if -i >= self.mleft:
                    yield(self.center + self.step * (-i))
                if i <= self.mright:
                    yield(self.center + self.step * i)
                



# def merge_with shifts(a: np.ndarray, max_shift: int, min_shift = None):
#     if min_shift is None:
#         min_shift = - max_shift
#     center = max_shift + min_
#     def gen_range(low, high):
#         yield(0)
#         if low < high:
#             for i in range(1, low):
#                 yield(i)
#                 yield(-i)

#             for 


#     def gen_shift(max):
#         tmp = itertools.product(gen_range(max_shift), repeat=a.shape[0]-1)
#         for t in tmp:
#             yield [0] + list(t)

#     def check(arr):
#         arr 

