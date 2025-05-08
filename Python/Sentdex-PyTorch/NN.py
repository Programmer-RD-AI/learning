#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# In[2]:


# Create Fully Connectied Network
class NN(nn.Module):
    def __init__(self,input_size=28*28,num_classes=10): # 28*28 img size, 10
        super().__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = NN()
x = torch.randn(64,28*28)
print(model(x).shape)


# In[3]:


# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# In[4]:


device


# In[5]:


# Hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 16
num_epochs = 125


# In[6]:


# Load data
train_dataset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


# In[7]:


# Init Network
model = NN(input_size,num_classes).to(device)


# In[8]:


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


# In[9]:


from tqdm import tqdm


# In[10]:


# Train network
for epoch in tqdm(range(num_epochs)):
    for index,(data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0],-1)
        # 32,784 (28*28)
        scores = model(data)
        loss = criterion(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# In[11]:


# Check accuracy on trsining and test to see how good our model
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print(f'{float(num_correct)/float(num_samples)*100}')
    model.train()


# In[12]:


check_accuracy(train_loader,model)


# In[13]:


check_accuracy(test_loader,model)


# In[ ]:




