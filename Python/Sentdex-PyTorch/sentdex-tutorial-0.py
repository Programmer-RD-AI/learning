#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])
print(x*y)


# In[2]:


x = torch.zeros([2,5])


# In[3]:


x


# In[4]:


x.shape


# In[6]:


x.reshape(10)


# In[8]:


y = torch.rand([2,5])


# In[ ]:




