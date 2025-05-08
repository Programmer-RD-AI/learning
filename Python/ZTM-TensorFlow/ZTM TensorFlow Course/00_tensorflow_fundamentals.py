#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# # In this notebook we are going to cover some of the most fundamental conceps of tensorflow (tensors)
# 
# More specifically, we're going to cover: 
# - Introduction to tensors
# - Getting information from tensors
# - Maniuplating tensors
# - Tensors & Numpy
# - Using @tf.funtion(Speed up python funtions)
# - Using GPUs with Tensorflow(or TPU)
# - Exercies

# ## Intro to tensors

# In[2]:


# Import tensorflow
import tensorflow as tf
print(f"Version - {tf.__version__}")


# In[3]:


# Create tensors with tf.constant()
scalar = tf.constant(7)
scalar


# In[4]:


# Check the number of dimentions of tensors (ndmin = number of dimensions)
scalar.ndim


# In[5]:


# Create a vector
vector = tf.constant([10,10])
vector
# shape 2 = num of stuff
# shape 1 = amount of arrays in there
# Not sure


# In[6]:


# Check the dimentions of our vector
vector.ndim


# In[7]:


# Create a matrix (more than 1 dimension)
matrix = tf.constant([[10,7],[7,10]])


# In[8]:


matrix


# In[9]:


matrix.ndim


# In[10]:


# Create a another matrix
another_matrix = tf.constant([[10.0,7.0],[3.0,6.0],[56.0,85.0]],dtype=tf.float16)


# In[11]:


another_matrix


# In[12]:


another_matrix.ndim # another of stuff in arrays


# In[13]:


tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
tensor


# In[14]:


tensor.ndim # amount of stuff in shape


# This is known as a rank 3 tensor (3-dimensions), however a tensor can have an arbitrary (unlimited) amount of dimensions.
# 
# For example, you might turn a series of images into tensors with shape (224, 224, 3, 32), where:
# 
# 224, 224 (the first 2 dimensions) are the height and width of the images in pixels.
# 3 is the number of colour channels of the image (red, green blue).
# 32 is the batch size (the number of images a neural network sees at any one time).
# All of the above variables we've created are actually tensors. But you may also hear them referred to as their different names (the ones we gave them):
# 
# scalar: a single number.
# vector: a number with direction (e.g. wind speed with direction).
# matrix: a 2-dimensional array of numbers.
# tensor: an n-dimensional arrary of numbers (where n can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector).
# To add to the confusion, the terms matrix and tensor are often used interchangably.
# 
# Going forward since we're using TensorFlow, everything we refer to and use will be tensors.

# ### Creating tensors with `tf.Variable()`

# In[15]:


# Create the same tensor with tf.Variable() as above
changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])


# In[16]:


changeable_tensor


# In[17]:


unchangeable_tensor


# In[18]:


# Lets try and change one of the elements in a changeable tensor
changeable_tensor


# In[19]:


# To numpy
changeable_tensor.numpy(),unchangeable_tensor.numpy()


# In[20]:


unchangeable_tensor.numpy()[0]


# In[21]:


# Try to assign
changeable_tensor[0].assign(7)
changeable_tensor


# In[22]:


# Unchngeable tensor

# unchangeable_tensor[0].assign(7)
# unchangeable_tensor 
# Error


# In[23]:


# Tf.contstant is better than tf.variable


# ### Creating random tensors
# 
# Random tensors are tensors of some abitrary size which contain random numbers

# In[24]:


num = 9+11+11+12+5+11+13+10+6+9+6+9+2+5+4+5+10+2.5+15
num/60


# In[25]:


# Create 2 random tensors
random_1 = tf.random.Generator.from_seed(42) # set seed for to reporoduce 
random_1 = random_1.normal(shape=(3,2))
random_2 = tf.random.Generator.from_seed(7)
random_2 = random_2.normal(shape=(3,2))
# Are they equal ? 
random_1 == random_2


# ### Shuffle data in tensors

# In[26]:


# pandas.sample(frac=1) *In Pandas*


# In[27]:


# Shuffle a tensor (valuable for when you want to shuffle the data so the inherent order doesnt effect learning)
not_shuffled = tf.constant([[10.0,7.0],[3.0,6.0],[56.0,85.0]])


# In[28]:


not_shuffled.ndim


# In[29]:


not_shuffled


# In[30]:


# Shuffle non-shuffled tensor
tf.random.set_seed(42)
tf.random.shuffle(not_shuffled,seed=42)


# It looks like if we want our shuffled tensor to be in the same order we have to use the gloabal and operation level random seed

# In[31]:


tf.random.set_seed(42) # gloabal random seed
tf.random.shuffle(not_shuffled,seed=42) # operation level random seed


# In[32]:


tf.random.shuffle(not_shuffled,seed=42) # operation level random seed


# ## Other ways to make tensors

# In[33]:


tf.zeros((1,25),tf.float32)


# In[34]:


tf.ones((1,25),tf.float32)


# ### Turn numpy arrays into tensors
# 
# The main difference between Numpy array and tensorflow tensors. the tensorflow tensors can be run on the GPU

# In[35]:


# Numpy array to tensorflow tensors
import numpy as np
numpy_A = np.arange(1,25,dtype=np.int32)
numpy_A
# Capaital for matrix or tensor
# Not capital for vector


# In[36]:


A = tf.constant(numpy_A,shape=(2,3,4)) # 2 * 3 * 4 = 24 (len(numpy_A))


# In[37]:


A


# In[38]:


A.ndim


# ### Getting information with tensors.
# 
# * Shape
# * Rank
# * Axis
# * Size

# In[39]:


tensor = tf.constant([0,1],dtype=tf.int8)


# In[40]:


# Shape
tensor.shape


# In[41]:


# Axis (1) = 1 | (2,3) = 2 | (3,4,5) = 3
tensor.ndim


# In[42]:


# Size (flaot/int)
tf.size(tensor)
# tf.size(tensor).dtype # get the exact size


# In[43]:


tensor[0].numpy()


# In[44]:


tensor[1].numpy()


# In[45]:


# Get various attributes of our tensor
print('Datatype of every element : ', tensor.dtype)
print('Number of dimnsion (rank) : ',tensor.ndim)
print('Shape of tensor : ',tensor.shape)
print('Number of elements : ',tf.size(tensor).numpy())


#  ### Indexing tensors
#  
#  Tensors can be indexed just like Python lists.

# In[46]:


some_list = [1,2,3,4]
some_list[:2]#,:2,:2,:2]


# In[47]:


some_list[:1]


# In[48]:


rank_4_tensor = tf.zeros([2,3,4,5])


# In[49]:


rank_4_tensor


# In[50]:


rank_2_tensor = tf.constant([[10,7],[3,4]])


# In[51]:


rank_2_tensor.shape


# In[52]:


rank_2_tensor.ndim


# In[53]:


rank_2_tensor


# In[54]:


some_list


# In[55]:


rank_2_tensor[:,-1].numpy()


# In[56]:


# Add in extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor[: , : , tf.newaxis]


# In[57]:


rank_3_tensor


# In[58]:


# Alternative
tf.expand_dims(rank_2_tensor,axis=-1) # Final axis


# In[59]:


tf.expand_dims(rank_2_tensor,axis=0)


# ### Manipulating tensors (tensor operations)
# 
# **Basic Operations** (+,-,*,/) (Python)

# In[60]:


# You can add values to a tensor using the addition operator
tensor = tf.constant([[10,7],[3,4]])
tensor+10


# In[61]:


tensor*2


# In[62]:


tensor/2


# In[63]:


tensor-5


# In[64]:


# We can use the tensorflow built-in funtion
tf.multiply(tensor,10)


# In[65]:


tf.add(tensor,5)


# In[66]:


tf.subtract(tensor,5)


# #### Matrix Multiplication
# 
# In machine learning, matrix multiplication is one of the mosr used tensor operations
# 
# There are 2 rules our tensors (or matrixces) need to fulfil :
# 
# 1. Inner dimensions must match
# 2. The resulting matrox has the shape of the inner dimensions

# In[67]:


# Matrix Multi
print(tensor)


# In[68]:


tf.matmul(tensor,tensor)


# In[69]:


tensor*tensor


# In[70]:


ex_1_tensor = tf.constant([
                           [1,2,1],
                           [0,1,0],
                           [2,3,4]
])
ex_2_tensor = tf.constant([
                           [2,5],
                           [6,7],
                           [1,8]
])


# In[71]:


ex_1_tensor.numpy()


# In[72]:


tf.matmul(ex_1_tensor,ex_2_tensor)


# In[73]:


# Matrix multi with Python operator "@"
tensor @ tensor


# In[74]:


tensor.shape


# In[75]:


# Create a tensor (3,2)
X = tf.constant([[1,2],[3,4],[5,6]])
y = tf.constant([[7,8],[9,10],[11,12]])


# In[76]:


X


# In[77]:


y


# In[78]:


# tf.matmul(X,y) # Error # The fliped one cant fit http://matrixmultiplication.xyz/


# In[79]:


X


# In[80]:


y.numpy().reshape(3,2)


# In[81]:


X @ y.numpy().reshape(2,3)


# **The Dot product**
# 
# Matrix multi is also reggered to as the dot product.
# 
# * `tf.matmul()`
# * `tf.tensordot()`
# * `@`

# In[82]:


# Perform the dot product on X and y (requires X or y to vbe transposed)
tf.tensordot(tf.transpose(X),y,axes=1)


# In[83]:


# Perfrom matrix multiplciation between X and y (transposed)
tf.matmul(X,tf.transpose(y))


# In[84]:


tf.matmul(X,tf.reshape(y,shape=(2,3)))


# In[85]:


print(f'Normal Y')
print(y)
print('\n')
print(f'Y reshaped to (2,3)')
print(tf.reshape(y,(2,3)))
print('\n')
print(f'Y trainsposed')
print(tf.transpose(y))


# In[86]:


tf.matmul(X,tf.transpose(y))


# In[87]:


# reshape is better


# ### Changing datatype of tensors

# In[88]:


# Create a new tensor with default datatype (float32)
B = tf.constant([1.7,7.4])
B.dtype


# In[89]:


C = tf.constant([7,10])
C.dtype


# In[90]:


tf.__version__


# In[91]:


# Change from float32 to float16 (reduced precision)
# 16 Bit Dtypes are Faster
D = tf.cast(B, dtype=tf.float16)
D


# In[92]:


E = tf.cast(C, dtype=tf.int8)
E


# ### Agreating tensor
# 
# Aggreating tensors = condensing then from multiple values down to smaller amount of values

# In[93]:


# Get the absoulute values
D = tf.constant([-7,10])
D


# In[94]:


# Get the absoulute values
abs(D) # turn all to positive


# In[95]:


tf.abs(D)


# Lets go through the following forms of aggresgation:
# * Get the max
# * Get the min
# * Get the mean
# * Get the sum

# In[96]:


max(D)


# In[97]:


min(D)


# In[98]:


tf.reduce_mean(D)


# In[99]:


sum(D)


# In[100]:


tf.reduce_min(D)


# In[101]:


tf.reduce_max(D)


# In[102]:


tf.reduce_mean(D)


# In[103]:


np.min(D)


# In[104]:


np.max(D)


# In[105]:


np.mean(D)


# In[106]:


np.min(D)


# In[107]:


# tf.math.reduce_std(D)


# In[108]:


import tensorflow_probability as tfp
tfp.stats.variance(D)


# In[109]:


# Find standard deviation
tf.math.reduce_std(tf.cast(D,dtype=tf.float16))


# In[110]:


tf.math.reduce_variance(tf.cast(E,tf.float16))


# In[111]:


# TypeError : turn to float or int try it - TypeError: Input must be either real or complex


# ### Find the position maximum and min

# In[112]:


# import numpy as np
# np.random.seed(42)


# In[113]:


# Create a new tensor for finsing positional min and max
tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
F


# In[114]:


F


# In[115]:


# Find the position max
tf.argmax(F)


# In[116]:


# Find the position min
tf.argmin(F)


# In[117]:


np.argmax(F)


# In[118]:


np.argmin(F)


# In[119]:


F[tf.argmax(F)]


# In[120]:


# Find the max value of F
tf.reduce_max(F)


# In[121]:


# Find the min value of F
tf.reduce_min(F)


# In[122]:


if F[tf.argmax(F)] == tf.reduce_max(F):
  print('Yes')
else:
  print('No')


# ### Squeezing a tensor (removing all singe dimensions)

# In[123]:


tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]),shape=(1,1,1,1,50))


# In[124]:


G


# In[125]:


G_s = tf.squeeze(G)


# ### One hot encode a tensor

# In[126]:


some_list = [0,1,2,3] # COULD BE RED GREEN BLUE PURPLE
# One hot encode our list of indices
tf.one_hot(some_list,len(some_list))


# In[127]:


# Specifiy custom vals
tf.one_hot(some_list,depth=4,on_value='Whats poppin',off_value='Hey !!')


# ### Squaring,log,square root

# In[128]:


H = tf.range(1,10)


# In[129]:


H


# In[130]:


# Square
tf.square(H)


# In[131]:


# Sqrt
tf.sqrt(tf.cast(H,tf.float16))


# In[132]:


# Log
tf.math.log(tf.cast(H,tf.float16))


# ### Tensors and numpy
# 
# TensorFlow interacts with Numpy arrays

# In[133]:


# Create a tensor directly from a Numpy Array
J = tf.constant(np.array([5,6,8]))


# In[134]:


J


# In[135]:


J.numpy()


# In[136]:


np.array(J)


# In[137]:


type(J)


# In[138]:


type(J.numpy())


# In[139]:


J = tf.constant([3])


# In[140]:


J.numpy()


# In[141]:


# The default types of each
numpy_J = tf.constant(np.array([3,7,10]))
tensorflow_J = tf.constant([3,7,10])


# In[142]:


numpy_J.dtype


# In[143]:


tensorflow_J.dtype


# ### Finding access to GPUs

# In[144]:


tf.test.is_gpu_available()


# In[145]:


tf.config.list_physical_devices()


# In[146]:


tf.test.is_built_with_cuda()


# In[147]:


tf.config.list_physical_devices('GPU')


# In[148]:


get_ipython().system('nvidia-smi')


# In[150]:


get_ipython().system('pip3 install wandb')


# In[ ]:




