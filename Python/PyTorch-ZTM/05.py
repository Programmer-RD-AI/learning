#!/usr/bin/env python
# coding: utf-8

# # What is Transfer learning?
# 
# Transfer learning is a technique in machine learning in which knowledge learned from a task is re-used in order to boost performance on a related task. For example, for image classification, knowledge gained while learning to recognize cars could be applied when trying to recognize trucks. Wikipedia
# 
# ## Why use transfer learning?
# 1. Can leverage an existing neural network architecture proven to work on problems similar to our own
# 2. Can leverage a woring network architecture which has already learned patterns on similar data to our own (so great performence with low data)

# In[1]:


import torch
import torchvision
torch.__version__,torchvision.__version__


# Now we've got the versions of torch and torchvision, we're after, let's import the code we've writte in previous section 

# In[2]:


# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    get_ipython().system('pip install -q torchinfo')
    from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular.going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    get_ipython().system('git clone https://github.com/mrdbourke/pytorch-deep-learning')
    get_ipython().system('mv pytorch-deep-learning/going_modular .')
    get_ipython().system('mv pytorch-deep-learning/data/pizza_steak_sushi_20_percent.zip ./data/05')
    get_ipython().system('unzip ./data/05/pizza_steak_sushi_20_percent.zip')
    # !rm -rf pytorch-deep-learning
    from going_modular.going_modular import data_setup, engine


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[4]:


get_ipython().system('nvidia-smi')


# ##  Get data
# 
# We need our pizza, steak, sushi data to build a transfer learning model

# In[5]:


# !unzip ./data/05/pizza_steak_sushi_20_percent.zip


# In[6]:


train_dir = "./data/05/train/"
test_dir = "./data/05/test/"


# ## Create Datasets and Dataloaders
# 
# Now've got some data, now we wanna turn it into PyTorch DataLoaders.
# 
# We can use `data_setup.py`and `create_dataloaders()`
# 
# Methods of transformations:
# 1. Manual
# 2. Automatically - the transforms are picked by pretrained model
# 
# When using a pretrained model, its important that the data that you pass through is transformed in the same way that the data was trained on

# In[7]:


# Pay attension when using pre trained models
from going_modular.going_modular import *


# ### Create transforms manually 

# In[8]:


from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# In[9]:


manual_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])


# In[10]:


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,test_dir,manual_transforms,32)


# ### Create transforms automatically

# In[11]:


import torchvision


# In[12]:


weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT # Default = best weight


# In[13]:


weights


# In[14]:


# Get the transforms used to create our pretrained weights
auto_transforms= weights.transforms()


# In[15]:


auto_transforms


# In[16]:


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,test_dir,auto_transforms,32)


# ## Getting a Pretrained model

# ### Which pretrained model should you use?
# 
# *Experiment, experiment, experiment!*
# 
# The whole idea of transfer learning is to take an already well performing model from a problem space similar to your own and then customize to your own problem.
# 
# Three things to consider:
# 1. Speed - how fast does it run?
# 2. Size - how big is it?
# 3. Performance - how accuracte is it?
# 
# Where would the model live?

# ### Setting up a pretrained model

# In[17]:


model = torchvision.models.efficientnet_b1(weights)


# In[18]:


# model


# In[19]:


model.avgpool


# In[20]:


# model.classifier[1] = torch.nn.Linear(1280,len(class_names),bias=True)


# ### Getting a summary of our model with torchinfo

# In[21]:


from torchinfo import summary


# In[22]:


summary(model,input_size=(1,3,224,224),col_names=['input_size','output_size','num_params','trainable'],col_width=20,row_settings=['var_names'])


# ### Freezing the base model and changing the output layer

# In[23]:


# Freeze all the base layers in EffNet
for param in model.features.parameters():
    param.requires_grad = False


# In[24]:


model.classifier[1] = nn.Linear(1280,len(class_names),bias=False)


# In[25]:


optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 5


# In[26]:


from going_modular.going_modular import model_builder,engine


# ## Train Model

# In[27]:


dict_metrics = engine.train(model, 
                     train_dataloader, 
                     test_dataloader, 
                     optimizer,
                     criterion,
                     epochs,
                     device
)


# In[28]:


import pandas as pd


# In[29]:


dict_pd = pd.DataFrame(dict_metrics)


# In[30]:


dict_pd.to_csv('./save/efficientnet_v2_m_fe.csv',index=False)


# In[33]:


pd.read_csv('./save/efficientnet_v2_m_fe.csv'),pd.read_csv('./save/efficientnet_v2_m_ft.csv')

