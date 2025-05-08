#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


get_ipython().system('pip3 install wandb')


# # Intro to NN classification with TensorFlow
# 
# In this notebook we're going to learn how to write a neural networks for classification problems.
# 
# A classification is where you try to classify as one thing or another thing
# 
# Few types of classification
# * Binary classification
# * Multiclass classification
# * Multilabel classifcation
# 

# ## Creating data to view and fit

# In[3]:


from sklearn.datasets import make_circles

# Make 1000 Example
n_samples = 1000
# Create circles 
X,y = make_circles(n_samples=n_samples,noise=0.03,random_state=42)


# In[4]:


X[:1]


# In[5]:


y[:1]


# ### Our data is a bit hard to understand lets visualize it.

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

circles = pd.DataFrame({'X0':X[:,0],'X1':X[:,1],'label':y})


# In[7]:


circles


# In[8]:


plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)


# ## Input and output shape

# In[9]:


# Check the shapes of our features and labels


# In[10]:


X.shape


# In[11]:


y.shape


# In[12]:


# Length
len(X)


# In[13]:


len(y)


# In[14]:


# View the first sample of X and y


# In[15]:


X[0],y[0]


# ## Steps in modelling
# 
# The steps in modelling with tensorflow typically: 
# 
# 1. Create or import a model
# 2. Compile the model
# 3. Fit the model
# 4. Evluate the model
# 5. Tweak
# 6. Evaluate...

# In[16]:


import tensorflow as tf
tf.__version__


# In[17]:


# Random Seed
tf.random.set_seed(42)

# Create the model using the sequential API
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.SGD())

# Fit
model_1.fit(X,y,epochs=5)


# In[18]:


# Train for longer
model_1.fit(X,y,epochs=200)


# In[19]:


model_1.evaluate(X,y)


# Since we are working ona  binary classification problem and out model if getting 50% accuracy it performing.
# 
# Lets add another layer

# In[20]:


tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(),metrics=['accuracy'])

model_2.fit(X,y,epochs=100,verbose=0)


# In[21]:


# Evluate
model_2.evaluate(X,y)


# ## Improving our model
# 
# Lets look into our bag of tricks to see how we can improve our model.
# 
# 1. Create a model - add more layers, increase the number of neurons, change the activation type.
# 2. Compileing a model - different optimizer, changing learning rate
# 3. Fiting a model - more epochs
# 

# In[22]:


tf.random.set_seed(42)

model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam())

model_3.fit(X,y,epochs=100,verbose=0)


# In[23]:


model_3.evaluate(X,y)


# To visualize our models predictions lets create  funtion `plot_decision_bounary()` 
# Ths funtion will: 
# 
# * Take in a trained mode, X and y.
# * Create a meshgrid of the different X values.
# * Make preds across the meshgrid
# * Plot the preds as well as a line between zones

# In[24]:


import numpy as np


# In[25]:


import numpy as np

def plot_decision_boundary(model, X, y):
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html
  
  # Make predictions using the trained model
  y_pred = model.predict(x_in)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())


# In[26]:


plot_decision_boundary(model_3,X,y)


# In[27]:


X[:,0].min() - 0.1


# In[28]:


X[:,0].min()


# In[29]:


X[:,0].max() + 0.1


# In[30]:


# Lets see if our model can be used for a regression problem
tf.random.set_seed(42)

X_reg = tf.range(0,1000,5)
y_reg = tf.range(100,1100,5)

X_reg_train = X_reg[:150]
y_reg_train = y_reg[:150]
X_reg_test = X_reg[150:]
y_reg_test = y_reg[150:]

# model_3.fit(X_reg_train,y_reg_train,epochs=100)


# In[31]:


X_reg_test


# In[32]:


X_reg_train


# In[33]:


y_reg_test


# In[34]:


y_reg_train


# oh.. wait.. we compiled our model_3 for a binary problem.
# 
# But.. we're now working in on a regresion problem, lets change the model

# In[35]:


tf.random.set_seed(42)

model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,metrics=['mae','mse'],optimizer=tf.keras.optimizers.Adam())


# In[36]:


# Lets see if our model can be used for a regression problem
tf.random.set_seed(42)

X_reg = tf.range(0,1000,5)
y_reg = tf.range(100,1100,5)

X_reg_train = X_reg[:150]
y_reg_train = y_reg[:150]
X_reg_test = X_reg[150:]
y_reg_test = y_reg[150:]

model_3.fit(X_reg_train,y_reg_train,epochs=100)


# In[37]:


# Make preds with our trained model
y_reg_preds = model_3.predict(X_reg_test)

# Plot the models predictions
plt.figure(figsize=(10,7))
plt.scatter(X_reg_train,y_reg_train,c='b',label='Train')
plt.scatter(X_reg_test,y_reg_test,c='g',label='Test')
plt.scatter(X_reg_test,y_reg_preds,c='r',label='Preds')
plt.legend();


# ## The missing pieace: Non-linearity

# In[38]:


# Set the random seed
tf.random.set_seed(42)

# Create the model
model_4 = tf.keras.Sequential([
  tf.keras.layers.Dense(1,activation=tf.keras.activations.linear)
])

# Compile
model_4.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam())

# Fit
history = model_4.fit(X,y,epochs=100,verbose=0)


# In[39]:


plot_decision_boundary(model_4,X,y)


# In[40]:


# Check out our data
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)


# In[41]:


model_4.evaluate(X,y)


# In[42]:


model_5 = tf.keras.Sequential([
  tf.keras.layers.Dense(1,activation=tf.keras.activations.linear)
])
model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model_5.fit(X,y,epochs=100)


# In[43]:


plot_decision_boundary(model_5,X,y)


# In[44]:


model_5.evaluate(X,y)


# In[45]:


model_6 = tf.keras.Sequential([
  tf.keras.layers.Dense(1,activation=tf.keras.activations.softmax)
])
model_6.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model_6.fit(X,y,epochs=100,verbose=0)
print(model_6.evaluate(X,y))
plot_decision_boundary(model_6,X,y)


# In[46]:


model_7 = tf.keras.Sequential([
  tf.keras.layers.Dense(1,activation=tf.keras.activations.tanh)
])
model_7.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model_7.fit(X,y,epochs=100,verbose=0)
print(model_7.evaluate(X,y))
plot_decision_boundary(model_7,X,y)


# In[47]:


model_8 = tf.keras.Sequential([
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu)
])
model_8.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=0.001))
model_8.fit(X,y,epochs=100,verbose=0)
print(model_8.evaluate(X,y))
plot_decision_boundary(model_8,X,y)


# In[48]:


model_9 = tf.keras.Sequential([
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(1)
])
model_9.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
history = model_9.fit(X,y,epochs=250)


# In[49]:


model_9.evaluate(X,y)


# In[50]:


plot_decision_boundary(model_9,X,y)


# In[51]:


model_8 = tf.keras.Sequential([
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(8,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid)
])
model_8.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=0.001))
model_8.fit(X,y,epochs=100)
print(model_8.evaluate(X,y))
plot_decision_boundary(model_8,X,y)


# In[52]:


model_8.evaluate(X,y)


# In[53]:


tf.random.set_seed(42)
model_7 = tf.keras.Sequential([
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(1,activation='sigmoid')
])
model_7.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
history = model_7.fit(X,y,epochs=100)


# In[54]:


plot_decision_boundary(model_8,X,y)


# In[55]:


print(model_8.evaluate(X,y))


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[57]:


# the combotniton of linear (straight lines) and non-linear (non-straigt lines) funtions is one of the key things of NN


# In[58]:


# Crete a toy tensor (similar to the data we pass into our models)
A = tf.cast(tf.range(-10,10),tf.float32)


# In[59]:


A


# In[60]:


# Visualize our toy tensor
plt.plot(A);


# In[61]:


# Lets start by replicating sigmoid
def sigmoid(x):
  return 1 / (1 + tf.math.exp(x))


# In[62]:


sigmoid(A)


# In[63]:


A


# In[64]:


# Plot our toy tensor transormed by sgmoid
plt.plot(sigmoid(A));


# In[65]:


def relu(x):
  return tf.maximum(0,x)


# In[66]:


plt.plot(relu(A))


# In[67]:


tf.keras.activations.linear(A)


# In[68]:


plt.plot(tf.keras.activations.linear(A))


# In[69]:


plt.plot(A)


# In[70]:


# ELU leakyrelu tabular


# ## Evaluating and improving our classification model

# In[71]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[72]:


# X_test,X_valid,y_test,y_valid = train_test_split(X_test,y_test,test_size=0.50)


# In[73]:


len(X_train),len(X_test)# ,len(X_valid)


# In[74]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[75]:


# Lets recreate a model to fit on the training data and evlauate on the testing
tf.random.set_seed(42)
model_8 = tf.keras.Sequential([
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(1,activation='sigmoid')
])
model_8.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=0.01))
history = model_8.fit(X_train,y_train,epochs=25)


# In[76]:


model_8.evaluate(X_test,y_test)


# In[77]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_8,X_train,y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_8,X_test,y_test)
plt.show();


# ## Plot the loss curves 

# In[78]:


history.history


# In[79]:


# Convert the history object into DF
pd.DataFrame(history.history)


# In[80]:


pd.DataFrame(history.history).plot()
plt.title('Model_8 Loss Curves')


# In[81]:


# Note for many problem the loss funtion going down is telling that the model is imrpoving


# ## Finding the best lr
# 
# To find the idea lr (the lr rate where the loss decreases the most during training) we're going to use the following steps:
# * A learning rate **callback** - you can think of a callback as an extra piece of funtionality, you can add to your *while training*.
# * Another model
# * A modified loss curve plot

# In[82]:


tf.random.set_seed(42)
model_9 = tf.keras.Sequential([
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(4,activation='relu'),
  tf.keras.layers.Dense(1,activation='sigmoid')
])
model_9.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam())
lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
history = model_9.fit(X_train,y_train,epochs=100,callbacks=[lr_schedular])


# In[83]:


pd.DataFrame(history.history).plot(figsize=(10,7),xlabel='epochs')


# In[84]:


# Plot the lr VS loss
lrs = 0.0001 * (10 ** (tf.range(100)/20))
plt.figure(figsize=(10,7))
plt.semilogx(lrs,history.history['loss'])
plt.xlabel('LR')
plt.ylabel('Loss')
plt.title('lr vs loss')


# In[85]:


0.01
0.1


# # lr's to use
# 
# - 1
# - 0.1
# - 0.01
# - 0.001
# - 0.0001
# - 0.00001

# In[86]:


model_8.evaluate(X_test,y_test)


# In[87]:


model_9 = tf.keras.Sequential([
  tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid)
])
model_9.compile(loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=0.02))
history = model_9.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=25)


# In[88]:


model_9.evaluate(X_test,y_test)


# In[89]:


model_9.evaluate(X_train,y_train)


# In[90]:


model_8.evaluate(X_test,y_test)


# In[91]:


model_8.evaluate(X_train,y_train)


# In[92]:


plot_decision_boundary(model_9,X_test,y_test)


# In[93]:


plot_decision_boundary(model_9,X_train,y_train)


# In[94]:


from sklearn.metrics import confusion_matrix


# In[95]:


# Precision (Highher)
# Accuracy (Higher)
# Confusion Matrix
# F1 Score (Lower)
# Recall ()
# Classification report (sklearn)


# In[96]:


model_9


# In[97]:


# Accuracy
loss,accuracy = model_8.evaluate(X_test,y_test)
print(f'Model (9) Loss (test data) : {loss}')
print(f'Model (9) Accuracy (test data) : {accuracy*100}%')


# ### Confusion matrix

# In[98]:


y_pred = model_8.predict(X_test)
from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test,y_pred)


# In[99]:


y_test[:1]


# In[100]:


y_pred[:1]


# In[101]:


# Opps.. Looks like our preds array has come out in prediction probaability form.. the standard output from sigmoid(or softmax) activation funtion


# In[102]:


y_pred = np.round(y_pred)


# In[103]:


pd.DataFrame(y_pred).value_counts()


# In[104]:


y_pred  = tf.squeeze(y_pred)


# In[105]:


confusion_matrix(y_test,y_pred)


# In[106]:


import seaborn as sns


# In[107]:


confusion_matrix(y_test,y_pred)


# How about we preetyy our confusion matrix

# In[108]:


# Note : the confusion matrix code we're going to write is a remix of sklearn `plot_confusion_matrix()`


# In[109]:


import itertools


# In[110]:


import seaborn as sns


# In[111]:


figsize = (10,10)

# Create confusion matrix
cm = confusion_matrix(y_test,y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis] # normaliize our confusion matrix
n_classess = cm.shape[0]
# Lets prettify it
fig,ax = plt.subplots(figsize=figsize)
# Create a matrix plot
cax = ax.matshow(cm,cmap=plt.cm.gray)
fig.colorbar(cax)
# Create classes
classes = False
if classes:
  labels = classes
else:
  labels = np.arange(cm.shape[0])
# Label the axis
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()
ax.yaxis.label.set_size(25)
ax.xaxis.label.set_size(25)
ax.set(title='Confusion Maxtrix',xlabel='Predicted Label',ylabel='True Label',xticks=np.arange(n_classess),yticks=np.arange(n_classess),xticklabels=labels,yticklabels=labels)
threshold = (cm.max() + cm.min()) / 2.
for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
  plt.text(j,i,f'{cm[i,j]} ({cm_norm[i,j]*100:.1f})',horizontalalignment='center',color='gray',size=25)


# In[112]:


cm_norm


# In[113]:


cm


# # Larger Problem (Classification (Multi))
# 
# When you have more than  classes as a option, its know as multi class classification
# 
# This means if you have 3 idfferebt classes, its mutli-class classification or even 100
# 
# To practise multi-class classification, we're going to build a NN to classify images of different items of cloathing

# In[114]:


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
# the data is already sorted into training and test sets for us
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()


# In[115]:


# Show the first training example
print(train_data[0])
print('-'*50)
print(train_labels[0])


# In[116]:


# Check the shape of a single example
train_data[0].shape,train_labels[0].shape


# In[117]:


# Plot a single sample
import matplotlib.pyplot as plt
plt.imshow(train_data[7]);


# In[118]:


# Check out samples label
train_labels[7]


# In[119]:


# Create a small list so we can index onto our training labels so they're human-readable
class_names = ['T-Shirt','Trouser','PullOver','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']


# In[120]:


len(class_names)


# In[121]:


# Plot an example iage and its label
index = 17
plt.imshow(train_data[index],cmap=plt.cm.binary)
plt.title(class_names[train_labels[index]])


# In[122]:


# Plot multiple random images of fashion MNIST
import random
plt.figure(figsize=(7,7))
for i in range(4):
  ax = plt.subplot(2,2,i+1)
  rand_index = random.choice(range(len(train_data)))
  plt.imshow(train_data[rand_index],cmap=plt.cm.binary)
  plt.title(class_names[train_labels[rand_index]])


# ## Building a multi-class classification model
# 
# For our multi-class classifcation model, we can use a similar architecture we will need to change some stuff.
# 
# * Input Shape - 28  X 28 (Shape of 1 IMG)
# * Output Shape - 10 (one per class of clothing)
# * Loss Funtion - tf.keras.losses.CategoricalCrossentropy()
#   * If your labels are one hot encoded use CategoricalCrossentropy
#   * Else use SparseCategoricalCrossentropy
# * Output Activation - Softmax (Not Sigmoid)

# In[123]:


# our data needs to be flattended (from 28*28 to None, 784)
flatten_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28))])
flatten_model.output_shape


# In[124]:


train_labels[:10]


# In[125]:


# Set random seed
tf.random.set_seed(42)

# Create the model
model_11 = tf.keras.Sequential(
  [
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
  ]
)
# CategoricalCrossentropy expects a OneHotEncoded Model but what we should use here is SparseCategoricalCrossentropy
model_11.compile(loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam())
non_norm_history = model_11.fit(train_data,
                                tf.one_hot(train_labels,depth=10),
                                epochs=10,
                                validation_data=(test_data,
                                                 tf.one_hot(test_labels,depth=10)))


# In[126]:


tf.one_hot(test_labels,depth=10)


# In[127]:


tf.one_hot(test_labels,depth=10) # 10 becuase the amount of classes in my multi classifcation data


# In[128]:


# Check the model summary
model_11.summary()


# In[129]:


# Check the min and max vals of training data
train_data.min(),train_data.max()


# In[131]:


model_11.layers


# In[132]:


model_11.summary()


# In[ ]:




