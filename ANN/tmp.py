# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:38:42 2018

@author: HP Zbook 15
"""

import numpy as np
a = [[1, 2, 3, 4],
     [2, 3, 4, 5],
     [3, 4, 5, 6],
     [4, 5, 6, 7]]
c = np.array(a)
c = c[:, :3]
b = np.reshape(c, [-1, 3])