#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)


# In[2]:


torch.device('cuda')


# In[3]:


get_ipython().system('nvidia-smi')


# ## Introduction to Tensors
# 
# ### Creating tensors

# In[4]:


# scalar

scalar = torch.tensor(7)
scalar


# In[5]:


scalar.ndim # finding the no. of dimentions


# In[6]:


scalar.item() # just get the data in the tensor


# In[8]:


# Vector
vector = torch.tensor([7,7])
vector


# In[9]:


vector.ndim


# In[10]:


vector.shape


# In[11]:


# Matrix
MATRIX = torch.tensor(
    [
        [7,8],
        [8,9]
    ]
)


# In[12]:


MATRIX.shape


# In[13]:


MATRIX.ndim


# In[14]:


# Tensor

TENSOR = torch.tensor([[[1,2,3,4],[5,6,5,4],[6,8,9,4]]])


# In[15]:


TENSOR


# In[16]:


TENSOR.shape # first dimension, second is row, third is element in row


# In[17]:


TENSOR.ndim


# In[18]:


TENSOR[0][2]


# ### Random Tensor
# 
# Why Random Tensors?
# This stf is important becz the way many neural networks learn is that they start with tensors full of random numbers and the adjust those random numbers to better represent the data
# 
# `Start with random Numbers -> Look at data -> update random numbers -> look at data
# Update the output data`

# In[19]:


rdm_tensor = torch.rand(3,4) # rows, columns


# In[20]:


rdm_tensor.shape


# In[21]:


rdm_tensor.ndim


# In[22]:


rdm_tensor


# In[23]:


# Create random tensor with similar shape to image tensor
random_image_size_tensor = torch.rand(size=(3,484,224)) # height, width, color
random_image_size_tensor.shape,random_image_size_tensor.ndim


# In[24]:


len(random_image_size_tensor[0]),len(random_image_size_tensor[1])


# ### Tensors with ones and zeros

# In[25]:


ones_tensor = torch.ones(3,4)
ones_tensor


# In[26]:


zeros_tensor = torch.zeros(3,4)
zeros_tensor


# In[27]:


rdm_tensor * zeros_tensor


# In[28]:


ones_tensor.dtype


# ### Range

# In[29]:


one_to_ten = torch.arange(start=1,end=11,step=1)


# In[30]:


one_to_ten


# In[31]:


ten_zeros = torch.zeros_like(input=one_to_ten)


# In[32]:


ten_zeros


# ### Tensor Datatypes

# In[33]:


# Float 32 tensor
float_32_tensor = torch.tensor([3.0,6.0,9.0],dtype=None,device='cuda',requires_grad=True)
float_32_tensor
# lower the bit no. the faster the data can be proccessed
# requires_grad means whther or not ot track the tensors gradient


# In[34]:


float_32_tensor.dtype


# In[35]:


float_16_tensor = float_32_tensor.type(torch.float16)


# In[36]:


float_16_tensor


# In[37]:


float_16_tensor * float_32_tensor


# ## Geting infor from tensor
# 
# Datatype - `tensor.dtype`
# 
# Shape - `tensor.shape`
# 
# Device - `tensor.device`

# In[38]:


# Create a tensor
some_tensor = torch.rand(3,4)
some_tensor


# In[39]:


print(some_tensor)
print(some_tensor.dtype)
print(some_tensor.size())
print(some_tensor.device)


# # Manipulating Tensors
# 
# Tensor operations include:
# 
# - Addition
# - Subtraction
# - Multiplication (Element wise)
# - Division
# - Matrix Multiplication

# In[40]:


# Addition

tensor = torch.tensor([1,2,3])
tensor + 10


# In[41]:


# Multiplication

tensor * 10


# In[42]:


# Subtraction

tensor - 10


# In[43]:


# Division

tensor / 1.5


# In[44]:


# Torch builtin functions

torch.mul(tensor,10)
torch.div(tensor,1.5)
torch.add(tensor,10)
torch.subtract(tensor,10)


# ### Matrix multiplication
# 
# Two main ways to perform multiplication in neural networks and deep learning:
#     
#     1. Element Wise Multiplication
#     2. Matrix Multiplication
#     
# 2 Rules when doing matrix multiplication 
# 
# 1. The inner dimmensions must match
# * (3,2) * (2,3) [This works]
# * (2,3) * (2,3) [Wont work]
# 
# 2. The out put matrix is of the outer dimensions
# * `(2,3) @ (3,2)` the output is (2,2)

# In[45]:


get_ipython().run_cell_magic('time', '', 'torch.matmul(tensor, tensor)\n')


# In[46]:


tensor*tensor


# In[47]:


tensor


# In[48]:


get_ipython().run_cell_magic('time', '', '# Matricx multiplication by hadnd\nvalue = 0\nfor i in range(len(tensor)):\n    value += tensor[i] * tensor[i]\n')


# ### Most common erros in DL
# 
# - Shape Errors

# In[49]:


tensor_A = torch.tensor([
        [1,2],
        [3,4],
        [5,6]
])

tensor_B = torch.tensor([
        [7,8],
        [8,11],
        [9,12]
])


# In[50]:


torch.matmul(tensor_A,tensor_B.view(2,3))


# In[51]:


tensor_A.T, tensor_A


# #### Transporse switches the axes of an tensor

# In[52]:


tensor_B.T,tensor_B


# ### Finding the Min Max Mean and SUm of Tensors

# In[53]:


x = torch.arange(0,100,10)


# In[54]:


x


# In[55]:


x.min()


# In[56]:


x.max()


# In[57]:


# requires a tensor of float or complex types
x.type(torch.float32).mean()


# In[58]:


x.sum()


# ### Finding the positional min and max

# In[59]:


x.argmax()


# In[60]:


x.argmin()


# ## Reshaping, stacking, squeezing, unsqueezing tensors
# 
# * Reshaping - reshapes an input tensor to a defined shape
# * View - Return a view of an input tensor of certain shape but keep the same memeory as the orgiiganl tensor
# * Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# * Squeezing - removes all `1` dimensions from a tensor
# * Unsqueeze - add a `1` dimension to a target tensor
# * Permute - Return a view of the input with dimensions permuted (swapped in a certain way)
# 

# In[61]:


x = torch.arange(1,10)
x


# In[62]:


x.shape


# In[63]:


x_reshape = x.reshape(1,9)


# In[64]:


x_reshape


# In[65]:


z = x.view(1,9)


# In[66]:


z.shape


# In[67]:


x.shape


# In[68]:


# view shares the same memory
z[0][0] = 5
z,x


# In[69]:


# Stack tensors on top
x_stacked = torch.stack([x,x,x,x],dim=1)


# In[70]:


x_stacked


# In[71]:


# torch.squeeze() remove all 1 dimensional shape


# In[72]:


x_reshape


# In[73]:


x_reshape.shape


# In[74]:


x_reshape.squeeze().shape


# In[75]:


x_squeezed = x_reshape.squeeze()


# In[76]:


x_squeezed.unsqueeze(dim=0)


# In[77]:


# premute - changes the dimensions of a tensors


# In[78]:


x_original = torch.rand(size=(224,224,3)) #height, width ,color channels


# In[79]:


torch.permute(x_original,(2,0,1)).shape, x_original.shape


# ### Indexing
# 
# Indexing with pytorch is similar with numpy

# In[80]:


# Create a tensor

x = torch.arange(1,10).reshape(1,3,3)


# In[81]:


x


# In[82]:


x[0]


# In[83]:


x[0][0]


# In[84]:


x[0][0][0]


# In[85]:


x[0][2][2]


# In[86]:


x[:,:,2]


# ## Pytorch tensors & Numpy
# 
# NumPy is a popular scientifica python numeical computing library
# 
# And becase of this Pytorch has functionalty to interact with it
# 
# * Data in Numpy, want in pytorch tensor -> `torch.from_numpy(ndarry)`
# * Pytorch tensors -> Numpy `torch.Tensor.numpy()`

# In[100]:


# NumPy array to tensor
import numpy as np

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array) # when converting from numpy, pyotrch reflects numpy's default datatype of float64
array,tensor


# In[101]:


tensor.dtype


# In[102]:


array = array +1


# In[105]:


tensor = torch.ones(7)
numpy_tensor = tensor.numpy()


# In[106]:


numpy_tensor,tensor


# ## Reproducibility (trying to take random out of random)
# 
# In short how a neural netowrk learns:
# 
# `start with random numbers -> tensor operations -> update random numbers to try and make them better representations`
# 
# To reduce the randomness in neural networks and PyTorch comes the concept of a **random seed**
# 
# Essentially what the random seed does is flavor the randomness. It makes it constant

# In[120]:


random_A = torch.rand(3,4)
random_B = torch.rand(3,4)

print(random_A)
print(random_B)
print(random_A == random_B)


# In[122]:


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_A = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
random_B = torch.rand(3,4)

print(random_A)
print(random_B)
print(random_A == random_B)


# ## Accessing GPU
# 
# GPUs= faster computation on numbers, thanks to CUDA + NVIDIA hardware + Pytorch working BTS 

# In[123]:


torch.cuda.is_available()


# In[ ]:




