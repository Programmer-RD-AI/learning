#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


# In[2]:


REBUILD_DATA = True


# In[3]:


class DogsVsCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat/"
    DOGS = "PetImages/Dog/"
    LABELS = {CATS:0,DOGS:1}
    training_data = []
    catcount = -1
    dogcount = -1
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    f = str(label) + str(f)
                    img = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save('training_data.npy',self.training_data)
        print(self.catcount)
        print(self.dogcount)


# In[4]:


if REBUILD_DATA:
    dvc = DogsVsCats()
    dvc.make_training_data()


# In[5]:


training_data = np.load('training_data.npy',allow_pickle=True)


# In[6]:


len(training_data)


# In[7]:


print(training_data[0][0])


# In[8]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[9]:


plt.imshow(training_data[1][0]/255.0,cmap='gray')
plt.show()


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[11]:


# https://www.coinmama.com/
# https://cex.io/
# https://www.coinfield.com/


# In[12]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 64, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(64, 128, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(128, 256, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        print(x.shape)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 1024) #flattening.
        self.fc2 = nn.Linear(1024, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


net = Net()
print(net)


# In[13]:


optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_funtion = nn.MSELoss()
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
VAL_PCT = 0.25
VAL_SIZE = int(len(X)*VAL_PCT)


# In[14]:


train_X = X[:-VAL_SIZE]
train_y = y[:-VAL_SIZE]
test_X = X[-VAL_SIZE:]
test_y = y[-VAL_SIZE:]


# In[15]:


print(len(test_X))


# In[16]:


print(len(train_X))


# In[17]:


BATCH_SIZE = 32


# In[18]:


# EPOCHS = 250
# for epoch in tqdm(range(EPOCHS)):
#     for i in range(0,len(train_X),BATCH_SIZE):
#         batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
#         batch_y = train_y[i:i+BATCH_SIZE]
#         net.zero_grad()
#         outputs = net(batch_X)
#         loss = loss_funtion(outputs,batch_y)
#         loss.backward()
#         optimizer.step()
# print(loss)


# In[19]:


# correct = 0
# total = 0
# net.eval()
# with torch.no_grad():
#     for i in tqdm(range(len(test_X))):
#         real_class = torch.argmax(test_y[i])
# #         print(real_class)
#         net_out = net(test_X[i].view(-1,1,50,50))
# #         print(net_out)
#         net_out = net_out[0]
#         predictied_class = torch.argmax(net_out)
#         if predictied_class == real_class:
#             correct += 1
#         total += 1
# print(round(correct/total,3))


# In[20]:


torch.cuda.is_available()


# In[21]:


get_ipython().system('nvidia-smi')


# In[22]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[23]:


device


# In[24]:


torch.cuda.device_count()


# In[25]:


# net.to(device)


# In[26]:


net = Net().to(device)


# In[ ]:


EPOCHS = 250


# In[ ]:


def train(net):
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    loss_funtion = nn.MSELoss()
    for epoch in tqdm(range(EPOCHS)):
        for i in range(0,len(train_X),BATCH_SIZE):
#         print(i,i+BATCH_SIZE)
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i+BATCH_SIZE]
            batch_X,batch_y = batch_X.to(device),batch_y.to(device)
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_funtion(outputs,batch_y)
            loss.backward()
            optimizer.step()
#         print(f'Epoch : {epoch}/{EPOCHS}, Loss : {loss.item()}')
    print(batch_X.shape)
    print(batch_y.shape)
    return outputs


# In[ ]:


train(net).shape


# In[ ]:


# CPU = 07:53
# GPU = 00:43


# In[ ]:


def test(net):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1,1,50,50).to(device))
            net_out = net_out[0]
            predictied_class = torch.argmax(net_out)
            if predictied_class == real_class:
                correct += 1
            total += 1
    print(round(correct/total,3))


# In[ ]:


test(net)


# In[ ]:


train(net)


# In[ ]:


test(net)


# In[ ]:


train(net)


# In[ ]:


test(net)


# In[ ]:


def fwd_pass(X,y,train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
        


# In[ ]:





# In[ ]:





# In[ ]:




