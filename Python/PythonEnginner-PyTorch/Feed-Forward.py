#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


import torch.nn as nn


# In[3]:


import torchvision


# In[4]:


import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# In[5]:


# device config
device = torch.device('cuda')


# In[6]:


# hyper parameters
input_size = 28*28 # img sizes
hidden_size = 100
num_classes = 10 # num of classes in dataset
num_epochs = 1000
batch_size = 100
learning_rate = 0.001


# In[7]:


# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())


# In[8]:


# Load MNIST dataset
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())


# In[9]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


# In[10]:


examples = iter(train_loader)


# In[11]:


samples,labels = next(examples)


# In[12]:


samples.shape # num of imgs # col channels # img size # img size


# In[25]:


for i in range(12):
    plt.subplot(2,3,i+1)tygjgyjftjtfhtuhdhty
    plt.imshow(samples[i][0],cmap='gray')
    plt.show()


# In[14]:


class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
        
    def forward(self,X):
        y_preds = self.l1(X)
        y_preds = self.relu(y_preds)
        y_preds = self.l2(y_preds)
        return y_preds


# In[15]:


model = NeuralNet(input_size,hidden_size,num_classes)


# In[16]:


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# In[17]:


# training loop
n_total_steps = len(train_loader)


# In[18]:


n_total_steps


# In[19]:


from tqdm import tqdm


# In[20]:


for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(100,28*28).to(device)
        labels = labels.to(device)
        # forward
        model.to(device)
        output = model(images)
        loss = criterion(output,labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch+1)
    print(loss.item())


# In[ ]:


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in tqdm(test_loader):
        images = images.reshape(100,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs,1)
        print(_)
        print(preds)
        break


# In[ ]:


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in tqdm(test_loader):
        images = images.reshape(100,28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _,preds = torch.max(outputs,1)
        print(n_samples)
        n_samples += labels.shape[0]
        print(n_samples)
        print((preds == labels).sum())
        print((preds == labels).sum().item())
        n_correct += (preds == labels).sum().item()
        print(n_correct)
        break
acc = 100.0 * n_correct / n_samples


# In[ ]:


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in tqdm(test_loader):
        images = images.reshape(100,28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _,preds = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (preds == labels).sum().item()
acc = 100 * n_correct / n_samples
print(acc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




