#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


N = 20
X = np.random.random(N) * 10 - 5

Y = 0.5 * X - 1 + np.random.randn(N)

plt.scatter(X,Y)


# In[ ]:




