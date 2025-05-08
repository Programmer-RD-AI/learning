#!/usr/bin/env python
# coding: utf-8

# In[70]:


# Design the model
# Make the loss and optimizr
# Training loop
# - forward pass : predict
# - backward pass : the loss with change its gradients
# - update weights : with the gradients of the loss funtion the optimizer will change its parameters (models)


# In[71]:


import torch


# In[72]:


import torch.nn as nn


# In[73]:


import numpy as np


# In[74]:


from sklearn import datasets


# In[75]:


from sklearn.preprocessing import StandardScaler


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


# 0) prepare the data
bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target
n_samples,n_features = X.shape
print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[78]:


sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


# In[79]:


y_train.shape


# In[80]:


y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


# In[81]:


y_train.shape


# In[97]:


# 1) model
class LR(nn.Module):
    def __init__(self,n_input,n_output=1):
        super().__init__()
        self.linear = nn.Linear(n_input,n_output)
        
    def forward(self,x):
        y_preds = torch.sigmoid(self.linear(x))
        return y_preds

model = LR(n_features)


# In[98]:


# 2) loss and otpimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


# In[99]:


# 3) training loop
num_epochs = 250
for epoch in range(num_epochs):
    y_preds = model(X_train)
    loss = criterion(y_preds,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    [w,b] = model.parameters()


# In[103]:


y_preds.round()[:10]


# In[104]:


y_test[:10]


# In[110]:


float(y_test.shape[0])


# In[111]:


y_preds.eq(y_test).sum()


# In[112]:


with torch.no_grad():
    y_preds = model(X_test)
    y_preds = y_preds.round()
    acc = y_preds.eq(y_test).sum()/float(y_test.shape[0])


# In[113]:


acc


# In[ ]:




