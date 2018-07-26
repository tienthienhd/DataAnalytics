# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:38:42 2018

@author: HP Zbook 15
"""

import numpy as np
import itertools


a = [[1, 2], [3.4, 4.3], ['a', 'b', 'c']]

def find_all_combinations(lists):
    res = set() #set so no duplicates
    def dfs(curr_index, curr_combination):
        if curr_index == len(lists): #base case - add to results and stop
            res.add(tuple(sorted(curr_combination)))
        else:
            for i in lists[curr_index]: #iterate through each val in cur index
                dfs(curr_index + 1, curr_combination + [i]) #add and repeat for next index
    dfs(0, [])
    return sorted(list(res))


b = list(itertools.product(a[0], a[1]))
for i in range(2, len(a)):
    b = list(itertools.product(b, a[i]))
    for j in range(len(b)):
        c = list(b[j][0])
        c.append(b[j][1])
        b[j] = c


#c = np.reshape(b, [len(b), len(a)])