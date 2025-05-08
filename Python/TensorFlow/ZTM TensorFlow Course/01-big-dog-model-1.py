#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install wandb')


# In[2]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Read in the insurance dataset
insurace = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
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


# In[3]:


import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"},name='750-20000-Final-2')#,project='test', entity='ranuga-d') # Sigmoid linear softmax Swish ReLU Tanh
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(4000,activation='swish'),
  tf.keras.layers.Dense(4000,activation='swish'),
  tf.keras.layers.Dense(4000,activation='swish'),
  tf.keras.layers.Dense(4000,activation='swish'),
  tf.keras.layers.Dense(4000,activation='swish'),
  tf.keras.layers.Dense(1),
])
model.compile(loss=tf.keras.losses.mae,metrics=['mae'],optimizer=tf.keras.optimizers.Adam())
model.fit(X_train_normal,y_train,validation_data=(ct.transform(X_test),y_test),callbacks=[WandbCallback(log_weights=True),tf.keras.callbacks.EarlyStopping(patience=25),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae',factor=0.2,patience=2,min_lr=0.001,verbose=2)],epochs=750)
model.save('/content/drive/MyDrive/Colab Notebooks/models/750-big-dog-model-final-v2.h5')

