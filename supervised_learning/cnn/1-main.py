#!/usr/bin/env python3

import numpy as np
pool_forward = __import__('1-pool_forward').pool_forward

np.random.seed(2)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
c = np.random.randint(2, 5)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 5, 2)).tolist()

X = np.random.uniform(0, 1, (m, h, w, c))
Y = pool_forward(X, (fh, fw), stride=(sh, sw), mode='max')
print(Y)
print(Y.shape)
