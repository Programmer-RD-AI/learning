#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# In[21]:


import pandas as pd
# train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
#                                   batch_size=-1, as_supervised=True)

# train_examples, train_labels = tfds.as_numpy(train_data)
# test_examples, test_labels = tfds.as_numpy(test_data)
data = pd.read_csv("./train.csv")
train_examples = np.array(data["text"])
train_labels = np.array(data["target"])
test_examples = np.array(data["text"])
test_labels = np.array(data["target"])


# In[22]:


print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))


# In[24]:


model = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)


# In[25]:


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))


# In[28]:


model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
history = model.fit(train_examples,
                    train_labels,
                    epochs=100,
                    batch_size=32,
                    validation_data=(test_examples, test_labels),
                    verbose=1)


# In[29]:


results = model.evaluate(test_examples, test_labels)

print(results)


# In[ ]:




