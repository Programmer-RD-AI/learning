#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.porter import *
from torch.nn import *
from torch.optim import *
import numpy as np
import pandas as pd
import torch,torchvision
import random
from tqdm import *
from torch.utils.data import Dataset,DataLoader
stemmer = PorterStemmer()


# In[2]:


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# In[3]:


tokenize('#100+')


# In[4]:


def stem(word):
    return stemmer.stem(word.lower())


# In[5]:


stem('organic')


# In[6]:


def bag_of_words(tokenized_words,all_words):
    tokenized_words = [stem(w) for w in tokenized_words]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_words:
            bag[idx] = 1.0
    return bag


# In[7]:


bag_of_words(['hi'],['how','hi'])


# In[8]:


data = pd.read_csv('./train.csv')


# In[9]:


X = data['text']
y = data['target']


# In[10]:


all_words = []
all_data = []
tags = []


# In[11]:


for X_batch,y_batch in tqdm(zip(X,y)):
    X_batch = tokenize(X_batch)
    new_X = []
    for Xb in X_batch:
        new_X.append(stem(Xb))
    all_words.extend(new_X)
    all_data.append((new_X,y_batch))
    tags.append(y_batch)


# In[12]:


np.random.shuffle(all_data)
np.random.shuffle(all_words)


# In[13]:


all_words = sorted(set(all_words))
tags = sorted(set(tags))


# In[14]:


np.random.shuffle(all_data)
np.random.shuffle(all_words)


# In[15]:


X = []
y = []


# In[16]:


for sentence,tag in tqdm(all_data):
    X.append(bag_of_words(sentence,all_words))
    y.append(tags.index(tag))


# In[17]:


from sklearn.model_selection import *


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.125,shuffle=False)


# In[19]:


device = 'cuda'


# In[20]:


torch.cuda.empty_cache()


# In[21]:


X_train = torch.from_numpy(np.array(X_train)).to(device).float()
y_train = torch.from_numpy(np.array(y_train)).to(device).to(device).long()
X_test = torch.from_numpy(np.array(X_test)).to(device).to(device).float()
y_test = torch.from_numpy(np.array(y_test)).to(device).to(device).long()


# In[22]:


def get_loss(model,X,y,criterion):
    preds = model(X)
    loss = criterion(preds.view(-1,1),y.view(-1,1))
    return loss.item()


# In[23]:


def get_accuracy(model,X,y):
    correcrt = 0
    total = 0
    preds = model(X)
    for pred,y_batch in zip(preds,y):
        pred = int(torch.round(pred))
        if pred == y_batch:
            correcrt += 1
        total += 1
    acc = round(correcrt/total,3)*100
    return acc


# In[24]:


class Model(Module):
    def __init__(self):
        super().__init__()
        self.activation = ReLU()
        self.iters = 10
        self.hidden = 512
        self.linear1 = Linear(len(all_words),self.hidden)
        self.linear2 = Linear(self.hidden,self.hidden)
        self.bn = BatchNorm1d(self.hidden)
        self.output = Linear(self.hidden,1)
    
    def forward(self,X):
        preds = self.linear1(X)
        for _ in range(self.iters):
            preds = self.activation(self.bn(self.linear2(preds)))
        preds = self.output(preds)
        return preds


# In[25]:


model = Model().to(device)


# In[26]:


criterion = MSELoss()


# In[27]:


model


# In[28]:


optimizer = Adam(model.parameters(),lr=0.001)


# In[29]:


epochs = 100


# In[30]:


batch_size = 8


# In[31]:


import wandb


# In[32]:


PROJECT_NAME = 'nlp-getting-started'


# In[33]:


torch.cuda.empty_cache()
wandb.init(project=PROJECT_NAME,name='baseline')
wandb.watch(model)
for _ in tqdm(range(epochs)):
    torch.cuda.empty_cache()
    for idx in range(0,len(X_train),batch_size):
        torch.cuda.empty_cache()
        X_batch = X_train[idx:idx+batch_size].to(device).float()
        y_batch = y_train[idx:idx+batch_size].to(device).float()
        preds = model(X_batch)
        loss = criterion(preds.view(-1,1),y_batch.view(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    wandb.log({'Loss':get_loss(model,X_train,y_train,criterion)})
    wandb.log({'Val Loss':get_loss(model,X_test,y_test,criterion)})
    wandb.log({'Acc':get_accuracy(model,X_train,y_train)})
    wandb.log({'Val Acc':get_accuracy(model,X_test,y_test)})
wandb.watch(model)
wandb.finish()
torch.cuda.empty_cache()


# In[ ]:




