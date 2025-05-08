#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


X = torch.randn(3,requires_grad=True)


# In[3]:


X


# In[4]:


y = X + 2


# In[5]:


y


# In[6]:


z = y*y*2


# In[7]:


z


# In[8]:


v = torch.tensor([0.1,1.0,0.001],dtype=torch.float32)


# In[9]:


z.backward(v)


# In[10]:


X.grad


# In[11]:


z


# In[12]:


X = torch.randn(3,requires_grad=True)


# In[13]:


X


# In[14]:


X.requires_grad_(False)


# In[15]:


X.detach()


# In[16]:


with torch.no_grad():
    print(X)


# In[17]:


weights = torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    print(model_output)
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()


# In[18]:


weights.grad.zero_()


# In[19]:


weights


# In[20]:


import torch


# In[21]:


x = torch.tensor(1.0)


# In[22]:


y = torch.tensor(2.0)


# In[23]:


w = torch.tensor(1.0,requires_grad=True)


# In[24]:


y_hat = w * x
loss = (y_hat - y)**2


# In[25]:


loss


# In[26]:


loss.backward(retain_graph=False)


# In[27]:


w.grad


# In[28]:


import numpy as np


# In[29]:


# f = w * x
# f = 2 * x
X = np.array([1,2,3,4],dtype=np.float32)


# In[30]:


Y = np.array([1*2,2*2,3*2,4*2],dtype=np.float32)


# In[31]:


w = 0.0


# In[32]:


# model preds
def forward(x):
    return w * x


# In[33]:


# loss = MSE
def loss(y,y_preds):
    return ((y_preds-y)**2).mean()


# In[34]:


# gradient
# MSE = 1/N * (w * x - y)**2
def gradient(x,y,y_preds):
    return np.dot(2*x,y_preds-y).mean()


# In[35]:


print(f'Preds before Training : f(5) = {forward(5):.3f}')


# In[36]:


learning_rate = 0.01
n_iters = 2500


# In[37]:


for epoch in range(n_iters):
    # preds = forward pass
    y_preds = forward(X)
    # loss
    l = loss(Y,y_preds)
    # gradients
    dw = gradient(X,Y,y_preds)
    # update weights
    w -= learning_rate * dw
    print(dw)
    print(f'Epoch : {epoch+1} Weight : {w:.3f} Loss = {l:.8f}')


# In[38]:


print(f'Preds before Training : f(5) = {forward(5):.3f}')


# In[39]:


import torch
# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model output
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 75

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)
    
    # calculate gradients
    l.backward()

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')


# In[40]:


import torch
import torch.nn as nn
# Here we replace the manually computed gradient with autograd

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model output
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w],lr=learning_rate)

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()
    
    # zero the gradients after updating
    optimizer.zero_grad()

    print(f'Prediction after training: f(5) = {forward(5).item():.3f}')


# In[41]:


# Design the model
# Make the loss and optimizr
# Training loop
# - forward pass : predict
# - backward pass : the loss with change its gradients
# - update weights : with the gradients of the loss funtion the optimizer will change its parameters (models)


# In[ ]:





# In[42]:


import torch
import torch.nn as nn
# Here we replace the manually computed gradient with autograd

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5],dtype=torch.float32)
n_samples,n_features = X.shape
input_size = n_features
output_size = n_features

class LinearRegreession(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        # define layers
        self.lin = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        return self.lin(x)

model = LinearRegreession(input_size,output_size)
    
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.1
n_iters = 250
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()
    
    # zero the gradients after updating
    optimizer.zero_grad()
    
    [w,b] = model.parameters()

    print(f'Prediction while training: f(5) = {model(X_test).item():.3f}')
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')


# In[ ]:





# In[ ]:





# In[ ]:




