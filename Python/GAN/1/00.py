#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# In[2]:


# Things to try:
# try larger network
# better normalization with batchnorm
# different lrs
# change architecture to a CNN


# In[3]:


IMG_SIZE = 28


# In[4]:


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3,32,5,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,64,5,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,5,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.dis2 = nn.Sequential(
            nn.Linear(3*28*28,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid(),
        )
     
    def forward(self, x,shape=False):
        x = x.view(-1,3,IMG_SIZE,IMG_SIZE)
        x = self.dis(x)
        if shape:
            print(x.shape)
        x = x.view(-1,3*28*28)
        x = self.dis2(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
 
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,IMG_SIZE*IMG_SIZE*3),
            nn.Tanh()
        )
 
    def forward(self, x):
        return self.gen(x)


# In[ ]:


# HYPERPARAMS
device = 'cuda'
lr = 3e-4
z_dim = 64 # 128,256,32 # I dont know what is z_dim but I know that that is a parameter at least :)
image_dim = 28*28*1 # 784
batch_size = 32
num_epochs = 25
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim,image_dim).to(device)
fixed_noise = torch.randn((batch_size,z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.4),(0.4))]
)
dataset = datasets.MNIST(root='dataset/',transform=transforms,download=True)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
opt_disc = optim.Adam(disc.parameters(),lr=lr)
opt_gen = optim.Adam(gen.parameters(),lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0


# In[ ]:


fixed_noise.shape


# In[ ]:


lossGs = []
lossDs = []


# In[8]:


for epoch in tqdm(range(num_epochs)):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.view(-1,28*28).to(device)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size,z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1) 
        lossD_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        output = disc(fake).view(-1)
        lossG = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        if batch_idx == 0:
            print(f'Epoch : [{epoch}/{num_epochs}] \n Loss D : {lossD:.4f}, Loss G : {lossG:.4f}')
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real = torchvision.utils.make_grid(data,normalize=True)
                writer_fake.add_image('Mnist Fake Imgs',img_grid_fake,global_step=step)
                writer_real.add_image('Mnist Real Imgs',img_grid_real,global_step=step)
                step += 1
                lossGs.append(lossG.item())
                lossDs.append(lossD.item())


# In[35]:


for epoch in tqdm(range(num_epochs)):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.view(-1,28*28).to(device)
        batch_size = real.shape[0]
        # Train Discriminator
        noise = torch.randn(batch_size,z_dim).to(device)
#         print(real.shape) # real image
#         print(noise.shape) # 
        fake = gen(noise)
#         print(fake.shape) # the fake image
        disc_real = disc(real).view(-1)
#         print(disc_real) # so like is this real or fake probability
        lossD_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1) 
#         print(disc_fake) # so like is this real or fake probability|
        lossD_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        # Train generator
        output = disc(fake).view(-1)
#         print(output) # so we are checking how good the gen here and in the above one we are checking for disc this is the same output
        lossG = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        if batch_idx == 0:
            print(f'Epoch : [{epoch}/{num_epochs}] \n Loss D : {lossD:.4f}, Loss G : {lossG:.4f}')
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real = torchvision.utils.make_grid(data,normalize=True)
                writer_fake.add_image('Mnist Fake Imgs',img_grid_fake,global_step=step)
                writer_real.add_image('Mnist Real Imgs',img_grid_real,global_step=step)
                step += 1
                lossGs.append(lossG.item())
                lossDs.append(lossD.item())


# In[36]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[37]:


get_ipython().run_line_magic('tensorboard', '--logdir runs')


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


plt.figure(figsize=(10,7))
plt.plot(lossGs)
plt.show()


# In[40]:


plt.figure(figsize=(10,7))
plt.plot(lossDs)
plt.show()


# In[41]:


# baseline = ok
# new model 1 = ok
# new model 2 = ok
# new model 3 = 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




