#!/usr/bin/env python
# coding: utf-8

# # Intro to Regression with Neural Networks in Tensorflow
# 
# There are many defintions for a regression problem but in our case, we are going to simpltfiy it. predicting a number with some other numbers

# In[ ]:


# Import Tensorflow
import tensorflow as tf
print(tf.__version__)


# ## Create data to view and fit

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Create features (Input)
X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
# Create labels (Output)
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0,21.0,24.0])


# In[ ]:


# Visualize
plt.scatter(X,y)


# In[ ]:


all(y == X + 10)


# In[ ]:


y == X + 10


# In[ ]:


X + 10


# ## Input and output shapes

# In[ ]:


# Create a demo tensor for our housing price prediction problem
house_info = tf.constant(['bedroom','bathroom','garage'])
house_price = tf.constant([939700])


# In[ ]:


house_info


# In[ ]:


house_price


# In[ ]:


X[0],y[0]


# In[ ]:


input_shape = X[0].shape
output_shape = y[0].shape


# In[ ]:


input_shape


# In[ ]:


output_shape


# In[ ]:


X[0].ndim


# In[ ]:


X[0],y[0]


# In[ ]:


# Turn our Numpy arrays to tensors
X = tf.cast(tf.constant(X),tf.float16)
y = tf.cast(tf.constant(y),tf.float16)


# In[ ]:


X


# In[ ]:


y


# In[ ]:


input_shape = X[0].shape
output_shape = y[0].shape


# In[ ]:


input_shape


# In[ ]:


output_shape


# In[ ]:


plt.scatter(X,y)


# ## Steps in modeling with tensorflow
# 
# 1. Creating a model - define the input and output layers, as well as the hidden layers also.
# 2. Compiling a model - define the loss funtion (in other words, the funtion which tells our model how wrong it is)
# 3. Fitting a model - letting the model find patter between X and y or Input and Output

# In[ ]:


tf.random.set_seed(42)
# Create a model using Squential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])
# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(), # or Adam()
              metrics=['accuracy']
              )
# MAE = how average the model is wrong (lower the better)
model.fit(X,y,epochs=5) # 50 is the best
# Epoch = Lap


# In[ ]:


X


# In[ ]:


y


# In[ ]:


y_pred = model.predict([17.0])


# In[ ]:


y_pred + 11


# ## Improve our model
# 
# We can improve our model by altering the steps we took to create a model
# 
# 1. Creating a model - Here we might add more layers (increase the number of neurons) within each of the hidden layers, change the activitation funtion of each layer
# 2. Compiling a model - here we might change the optimization funtion or perhaps *learning rate*
# 3. Fit a model - add more epochs. or more data

# In[ ]:


# Lets rebuild our model

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

# Fit the model
model.fit(X,y,epochs=100)


# In[ ]:


# Data
X,y


# In[ ]:


model.predict([17])


# In[ ]:


model = tf.keras.Sequential([
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(2500,activation='relu'),
  tf.keras.layers.Dense(1)
])
model.compile(metrics=['mae'],optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.mae)
model.fit(X,y,epochs=100)


# In[ ]:


round(int(model.predict([17])[0]))


# In[ ]:


model = tf.keras.Sequential([
  tf.keras.layers.Dense(100,activation='relu'),
  tf.keras.layers.Dense(1)
])
model.compile(metrics=['mae'],optimizer=tf.keras.optimizers.SGD(),loss=tf.keras.losses.mae)
model.fit(X,y,epochs=100)


# In[ ]:


# Data


# In[ ]:


X


# In[ ]:


y


# In[ ]:


model.predict([17])


# In[ ]:


plt.scatter(X,y)


# In[ ]:


X = np.arange(-100, 100, 4)
X
y = np.arange(-90, 110, 4)
y


# In[ ]:


# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

len(X_train), len(X_test)

# or train_test_split


# In[ ]:


plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(X_train, y_train, c='b', label='Training data')
# Plot test data in green
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Show the legend
plt.legend();


# In[ ]:


# Lets have a look at how to build a NN for our data
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100,activation=None),
  tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.mae,metrics=['mae'])
model.fit(X_train,y_train,epochs=100)


# In[ ]:


y_preds = model.predict(X_test)


# In[ ]:


# Lets create a model which builds automatically by defining the input_shape argument in the first layer
tf.random.set_seed(42)
# 1. Create a model (Same)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(62500,input_shape=[1],name='input_layer'),
  tf.keras.layers.Dense(1,name='output_layer')
],name='model_1')

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])


# In[ ]:


model.summary()


# * Total params - total number of parameters in the model. how many nerous * 2
# * Trainable parameter - these are the params (patterns) the model can update as it trains.
# * Non trainable params - these params arent updated durning traning (this changes when you get a pre trained model using tranfer learning)

# In[ ]:


y


# In[ ]:


# Dense = fully connect layer


# In[ ]:


model.fit(X_train,y_train,epochs=250)


# In[ ]:


model.predict(X_test)


# In[ ]:


y_test


# In[ ]:


np.round(model.predict(X_test))


# In[ ]:


np.round(model.predict(X_test)) == np.array([y_test])


# In[ ]:


np.round(model.predict(X_test))


# In[ ]:


np.array([y_test])


# In[ ]:


y_test


# In[ ]:


# Get a summary of our model
model.summary()


# In[ ]:


from tensorflow.keras.utils import plot_model


# In[ ]:


plot_model(model=model,show_shapes=True)


# ### Visuallizing our models predictions
# 
# To visualize predictions, its good to idea to plot them against the ground truth labels
# 
# often like `y_test` or `y_true` VS `y_pred` (the real data VS the predictions)

# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_test


# In[ ]:


y_pred


# In[ ]:


# Lets create your plotting funtion
def plot_preds(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,preds=y_pred):
  """
  Plots training data and test data and compares data to the groud truth labels
  """
  plt.figure(figsize=(10,7))
  # Plot training data in blue
  plt.scatter(train_data,train_labels,c='b',label='Training Data')
  # Plot testing data in green
  plt.scatter(test_data,test_labels,c='g',label='Testing Data')
  # Plot models preds in red
  plt.scatter(test_data,preds,c='r',label='Preds')
  # Show the legend
  plt.legend();


# In[ ]:


plot_preds(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,preds=y_pred)


# ## Evaluating our models predictions with regression evaluation metrics

# In[ ]:


def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(tf.squeeze(y_test),
                                        tf.squeeze(y_pred))
  
def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.metrics.mean_squared_error(tf.squeeze(y_test),
                                        tf.squeeze(y_pred))


# In[ ]:


# Set random seed
tf.random.set_seed(42)

# Replicate original model
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model
model_1.fit(X_train, y_train, epochs=100)


# In[ ]:


# Set random seed
tf.random.set_seed(42)

# Replicate original model
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model
model_1.fit(X_train, y_train, epochs=100)


# In[ ]:


# Make and plot predictions for model_1
y_preds_1 = model_1.predict(X_test)
plot_preds(preds=y_preds_1)


# In[ ]:


# Set random seed
tf.random.set_seed(42)

# Replicate model_1 and add an extra layer
model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1) # add a second layer
])

# Compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model
model_2.fit(X_train, y_train, epochs=100, verbose=0) # set verbose to 0 for less output


# In[ ]:


# Make and plot predictions for model_2
y_preds_2 = model_2.predict(X_test)
plot_preds(preds=y_preds_2)


# In[ ]:


# Set random seed
tf.random.set_seed(42)

# Replicate model_2
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# Compile the model
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model (this time for 500 epochs, not 100)
model_3.fit(X_train, y_train, epochs=500, verbose=0) # set verbose to 0 for less output


# In[ ]:


# Make and plot predictions for model_3
y_preds_3 = model_3.predict(X_test)
plot_preds(preds=y_preds_3)


# In[ ]:


y_pred


# In[ ]:


y_preds_1


# In[ ]:


y_preds_2


# In[ ]:


y_preds_3


# In[ ]:


mae_1 = mae(y_test,model_1.predict(X_test))
mse_1 = mse(y_test,model_1.predict(X_test))

mae_2 = mae(y_test,model_2.predict(X_test))
mse_2 = mse(y_test,model_2.predict(X_test))

mae_3 = mae(y_test,model_3.predict(X_test))
mse_3 = mse(y_test,model_3.predict(X_test))


# In[ ]:


model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
                 ["model_2", mae_2.numpy(), mse_2.numpy()],
                 ["model_3", mae_3.numpy(), mae_3.numpy()]]
import pandas as pd
all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
all_results


# Tracking your experiments
# One really good habit to get into is tracking your modelling experiments to see which perform better than others.
# 
# We've done a simple version of this above (keeping the results in different variables).
# 
# 📖 Resource: But as you build more models, you'll want to look into using tools such as:
# 
# TensorBoard - a component of the TensorFlow library to help track modelling experiments (we'll see this later).
# Weights & Biases - a tool for tracking all kinds of machine learning experiments (the good news for Weights & Biases is it plugs into TensorBoard).

# ## Saving a model
# 
# Saving our model allows us to use them outside of Google Colab, and use them in a web app or a mobile app.

# In[ ]:


# Saved Model Format
# HDF5 Format


# In[ ]:


model_2.save('2-layers--100-epochs')


# In[ ]:


model_2.save('2-layers--100-epochs.h5')


# In[ ]:


tf.keras.models.save_model(model_2,'./model.h5') # I think this is better


# In[ ]:


# .h5 way is the best


# In[ ]:


model_2_loaded = tf.keras.models.load_model('2-layers--100-epochs.h5')


# In[ ]:


model_2_loaded.summary()


# In[ ]:


model_2.summary()


# In[ ]:


model_2_loaded.evaluate(X_test,y_test)


# In[ ]:


model_2.evaluate(X_test,y_test)


# In[ ]:


model_2.evaluate(X_test,y_test)


# In[ ]:


model_2_loaded.evaluate(X_test,y_test)


# In[ ]:


# Download Files (Code)
from google.colab import files
files.download('/content/2-layers--100-epochs.h5')


# In[ ]:


# Google Colab to Google Drive
get_ipython().system('cp /content/2-layers--100-epochs.h5 /content/drive/MyDrive/Colab\\ Notebooks/models/01')


# # A larger Example

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[ ]:


# Read in the insurance dataset
insurace = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')


# In[ ]:


insurace


# In[ ]:


len(insurace)


# In[ ]:


insurace.dropna(inplace=True)


# In[ ]:


len(insurace)


# In[ ]:


def object_to_int(data,column):
  index = -1
  all_info = []
  info_dict = {}
  for info in data[column]:
    if info not in info_dict:
      index = index + 1
      info_dict[info] = index
  for info in data[column]:
    all_info.append(info_dict[info])
  return (index,all_info,info_dict)


# In[ ]:


insurace.head()


# In[ ]:


insurace.dtypes


# In[ ]:


insurace['bmi'] = insurace['bmi'].astype(int)
insurace['charges'] = insurace['charges'].astype(int)


# In[ ]:


insurace.dtypes


# In[ ]:


sex_info = object_to_int(insurace,'sex')
insurace['sex'] = sex_info[1]
smoker_info = object_to_int(insurace,'smoker')
insurace['smoker'] = smoker_info[1]
region_info = object_to_int(insurace,'region')
insurace['region'] = region_info[1]
# or pd.get_dummies(insurace)


# In[ ]:


insurace.dtypes


# In[ ]:


insurace.head()


# In[ ]:


insurace['region'].value_counts()


# In[ ]:


insurace['smoker'].value_counts()


# In[ ]:


insurace['sex'].value_counts()


# In[ ]:


insurace.head()


# In[ ]:


X = insurace.drop('charges',axis=1)
y = insurace['charges']


# In[ ]:


from sklearn.metrics import *
from sklearn.model_selection import *


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


get_ipython().system('pip3 install wandb')


# In[ ]:


import shutil
try:
  shutil.rmtree('/content/wandb/')
except:
  shutil.rmtree('/content/wandb')


# In[ ]:


# import wandb
# from wandb.keras import WandbCallback
# wandb.init(config={"hyper": "parameter"},name='500-20000-Final-1')#,project='test', entity='ranuga-d') # Sigmoid linear softmax Swish ReLU Tanh
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(1),
#   tf.keras.layers.Dense(4000,activation='swish'),
#   tf.keras.layers.Dense(4000,activation='swish'),
#   tf.keras.layers.Dense(4000,activation='swish'),
#   tf.keras.layers.Dense(4000,activation='swish'),
#   tf.keras.layers.Dense(4000,activation='swish'),
#   tf.keras.layers.Dense(1,activation='softmax'),
#   tf.keras.layers.Dense(1),
# ])
# model.compile(loss=tf.keras.losses.mae,metrics=['mae'],optimizer=tf.keras.optimizers.Adam())
# model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[WandbCallback(log_weights=True),tf.keras.callbacks.EarlyStopping(patience=5),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae',factor=0.2,patience=2,min_lr=0.001,verbose=2)],epochs=500)
# model.save('/content/drive/MyDrive/Colab Notebooks/models/500-big-dog-model.h5')
# wandb.finish()


# In[ ]:


# Layers = 5
# Epochs = 500 or 750
# learning rate = 0.001
# total neurons = 20000
# activation type = Swish


# In[ ]:


# model.save('/content/drive/MyDrive/Colab Notebooks/models/750-big-dog-model.h5')


# In[ ]:


# Modelling Course
tf.random.set_seed(42)

# Create a model
insurance_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])
# Compile
insurance_model.compile(loss=tf.keras.losses.mae,metrics=['mae'],optimizer=tf.keras.optimizers.Adam())
# Fit
insurance_model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100)


# In[ ]:


insurance_model.evaluate(X_test,y_test)


# In[ ]:


y_test.median(),y_test.mean()


# In[ ]:


y_pred = insurance_model.predict(X_test)


# In[ ]:


len(y_pred)


# In[ ]:


len(X_test)


# In[ ]:


len(y_pred[0])


# Right now the model is not the best kets try and improve our model
# 
# To (try) to improve our model, we'll run 2 experiments.
# 
# 1. Add an extra layer with more hidden unis
# 2. Train for longer

# In[ ]:


tf.random.set_seed(42)

# Create the model
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])
insurance_model_2.compile(loss=tf.keras.losses.mae,metrics=['mae'],optimizer=tf.keras.optimizers.Adam())
insurance_model_2.fit(X_train,y_train,epochs=100)


# In[ ]:


insurance_model_2.evaluate(X_test,y_test)


# In[ ]:


y_test.mean()


# In[ ]:


np.array(insurance_model_2.predict(X_test)).mean()


# In[ ]:


tf.random.set_seed(42)

# Create the model
insurance_model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1),
])
insurance_model_3.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])
history = insurance_model_3.fit(X_train,y_train,epochs=2500,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae',patience=25,verbose=2),tf.keras.callbacks.ReduceLROnPlateau(monitor='mae',factor=0.2,patience=2,min_lr=0.001,verbose=2)])


# In[ ]:


insurance_model_3.evaluate(X_test,y_test)


# In[ ]:


insurance_model_2.evaluate(X_test,y_test)


# In[ ]:


insurance_model.evaluate(X_test,y_test)


# In[ ]:


# Plot history (also know as loss curve or a training curve)


# In[ ]:


pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')


# > ? How long should you train for ? 
# 
# It depends... It depends on the problem. alot of people asked this question. so tensorflow has gave a solution EarlyStoppingCallbacks, which is a tensorflow componet you can add to your model to stop training once it stop training.

# In[ ]:


get_ipython().system('nvidia-smi')


# ## Preproccessing data (normalization and standardisation)
# 
# In terms of scaling values, NN tend to prefer normalization.
# 
# Try both and find what is better.
# 
# 

# In[1]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Read in the insurance dataset
insurace = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')


# In[2]:


insurace.head()


# In[3]:


# To preapre our data, we can borrow some classes from Sklearn


# In[4]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
# Create a coloum transformer
ct = make_column_transformer(
    (MinMaxScaler(), ['age','bmi','children']), # turn all values in this coloum between 0 and 1
    (OneHotEncoder(handle_unknown='ignore'), ['sex','smoker','region'])
)

# Create X and y
X = insurace.drop('charges',axis=1)
y = insurace['charges']

# Split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# Fit the coloum transformer to training data
ct.fit(X_train)

# Transform training and data with (Min max scaler) and One hot encoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)


# In[5]:


X_train_normal[0]


# In[6]:


X_train.loc[0]


# In[7]:


# Shape of the data
X_train.shape,X_train_normal.shape


# In[8]:


# test_data = np.array([19,'male',30,0,'no','southwest'])
# ct.transform(test_data)


# In[9]:


np.array(X_train.loc[0])


# In[10]:


# test_data


# In[11]:


ct.transform(X_train)


# In[12]:


test_data = X_train.iloc[0]


# In[13]:


# ct.transform([test_data])


# In[14]:


np.array(test_data)


# In[15]:


X_train


# In[16]:


X_train_normal


# In[17]:


np.array(X_train.iloc[0])


# In[18]:


test_data = pd.DataFrame(np.array([[24, 'male', 23.655, 0, 'no', 'northwest']]),columns=['age','sex','bmi','children','smoker','region'])


# In[19]:


test_data


# In[20]:


test_data


# In[21]:


ct.transform(test_data)


# In[22]:


# Beautiful! our data is normalized and one hot encoded.


# In[24]:


get_ipython().system('pip install wandb')


# In[25]:


import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"},name='25000')#,project='test', entity='ranuga-d') # Sigmoid linear softmax Swish ReLU Tanh
model = tf.keras.Sequential([
  tf.keras.layers.Dense(25000,activation='swish'),
  tf.keras.layers.Dense(1),
])
model.compile(loss=tf.keras.losses.mae,metrics=['mae'],optimizer=tf.keras.optimizers.Adam(lr=1))
model.fit(X_train_normal,y_train,validation_data=(ct.transform(X_test),y_test),callbacks=[WandbCallback(log_weights=True),tf.keras.callbacks.EarlyStopping(patience=25)],epochs=250)
model.save('/content/drive/MyDrive/Colab Notebooks/models/750-big-dog-model-final-v2.h5')


# In[30]:


# Build a NN model to fit on our normalized data (Course)
tf.random.set_seed(42)

# Create the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.mae,metrics=['mae','mse'])
model.fit(X_train_normal,y_train,epochs=100)


# In[33]:


# Evluate the model
model.evaluate(X_test_normal,y_test)


# In[ ]:




