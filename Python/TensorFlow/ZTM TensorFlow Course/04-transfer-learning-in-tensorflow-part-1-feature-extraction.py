#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning with Tensorflow Part 1 : Feature Extraction
# 
# Transfer learning is leveraging a working models exisiting architecture and learned patterns
# 
# Benifits
# - Can leverage an exsting NN architecture proven to work
# - Can leverage a working NN which has already learned the patterns and we can adapt the patterns our problem

# In[ ]:


get_ipython().system('nvidia-smi')


# ## Downloading and becomming one with the data

# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip')


# In[ ]:


get_ipython().system('unzip 10_food_classes_10_percent.zip')


# In[ ]:


import os


# In[ ]:


# How many images in each folder ? 
# Walk through 10% data dir and list number of files
for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# ## Create data loaders (preparing the data)
# 
# We'll use the ImageDataGenerator class to load our images in batches

# In[ ]:


# Setup data inputs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SHAPE = (224,224)
BATCH_SIZE = 32
train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"
train_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)
print('Training Images')
train_data_10_percent = train_datagen.flow_from_directory(train_dir,target_size=IMAGE_SHAPE)
print('Testing Images')
test_data_10_percent = test_datagen.flow_from_directory(test_dir,target_size=IMAGE_SHAPE)


# ## Setting up callbacks (things to run while our model trains)
# 
# Callbacks are extra funtionality you can add to your model to be performed during or after training. Some of the most popular callbacks:
# 
# * Tracking experiments (Tensorboard)
# * Model check point (ModelCheckpoint)
# * Stop overfitting (EarlyStopping)

# In[ ]:


import tensorflow as tf


# In[ ]:


# Create TensorBoard callback (functionized beause we need to create a new one for each model)
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


# ## Creating models using tensorflow hub
# 
# So in the past we have used tensorflow to make models from sctrach but now we are going to do a same like proccess but the architecture is coming from https://tfhub.dev/
# 
# Browsing the Tensorflow hub page we found the following model (feature vector)
# https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1

# In[ ]:


# Lets compare the following 2 models
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
efficentnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"


# In[ ]:


# Import dependencies
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


# In[ ]:


# Lets make a create_model funtion to create a model from a url
def create_model(model_url,num_classes=10):
  feature_extractor_layer = hub.KerasLayer(model_url,
                                          trainable=False, # remove all of the trained patterns
                                          name="feature_extraction_layer",
                                          input_shape=(224,224,3)
                                           )
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(num_classes,activation='softmax')
  ])
  return model


# In[ ]:


# Create and testing resnet tensorflow hub feature extraction model
resnet_model = create_model(resnet_url,num_classes=train_data_10_percent.num_classes)


# In[ ]:


resnet_model.summary()


# In[ ]:


# Compiel our resnet moel
resnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])


# In[ ]:


resnet_model.fit(train_data_10_percent,epochs=5,callbacks=[create_tensorboard_callback('tensorflow_hub','resnet50v2')],validation_data=test_data_10_percent,)


# Wow ! 
# 
# That is incrediable. Our transfer learning feature extractor model out performs ALL of the previous models we built by hand. and with a quicker time and with only 10% of the training examples

# In[ ]:


import pandas as pd
preds = pd.DataFrame(resnet_model.evaluate(test_data_10_percent))


# In[ ]:


preds


# ## Creating and testing EfficentNetB0 Tensorflow hub feature extraction model

# In[ ]:


efficent_model = create_model(efficentnet_url)
efficent_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
efficent_history = efficent_model.fit(train_data_10_percent,epochs=5,callbacks=[create_tensorboard_callback('tensorflow_hub','efficentnetb0')],validation_data=test_data_10_percent,)


# In[ ]:


pd.DataFrame(efficent_model.evaluate(test_data_10_percent))


# In[ ]:


efficent_model.summary()


# In[ ]:


resnet_model.summary()


# ## Different types of transger learning
# 
# * "AS if" transfer learning - using an existing model with no change what so ever
# * "Feature Extraction" transfer learning - us the pre learned pattern of an existing model
# * "Find tuning" transfer learning - use the pretrane model and fine tune the underlying layers

# ## Comparing our models results using Tensorboard
# 
# > Warning : When you upload things to tensorboard.dev your experiments are public so if you are running private experiments (things you dont want others to see) dont use tensorboard !!!!

# In[ ]:


# Upload TensorBoard dev records
get_ipython().system('tensorboard dev upload --logdir ./tensorflow_hub/    --name "EfficientNetB0 vs. ResNet50V2"    --description "Comparing two different TF Hub feature extraction model architectures using 10% of the training data"    --one_shot')


# In[ ]:


# Link https://tensorboard.dev/experiment/dQBrpdwIRgS2qI0Andv8Yg/#scalars


# In[ ]:


get_ipython().system('tensorboard dev list')


# In[ ]:


# Delete
# !tensorboard dev delete --experiment_id 


# In[ ]:


get_ipython().system('tensorboard dev list')


# In[ ]:




