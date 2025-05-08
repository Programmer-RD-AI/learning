#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.nn import *
import torch.nn as nn
import wandb
from torch.optim import *
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
PROJECT_NAME = 'Learning-DCGAN'


# In[2]:


class Discriminator(Module):
    def __init__(self,channels_img,features_d):
        super().__init__()
        self.disc = Sequential(
            Conv2d(channels_img,features_d,kernel_size=4,stride=2,padding=1),
            LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            Conv2d(features_d*8,1,4,stride=2,padding=0),
            Sigmoid()
        )
    
    def _block(self,in_channels,out_channels,kernal_size,stride,padding):
        return Sequential(
            Conv2d(
            in_channels,out_channels,kernal_size,stride,padding,bias=False),
            BatchNorm2d(out_channels),
            LeakyReLU()
        )

    def forward(self,X):
        return self.disc(X)


# In[3]:


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# In[4]:


def initalize_weights(model):
    for m in model.modules():
        if isinstance(m,(Conv2d,ConvTranspose2d,BatchNorm2d)):
            init.normal_(m.weight.data,0.0,0.02)


# In[5]:


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# In[6]:


test()


# In[7]:


device = 'cuda'


# In[8]:


LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64


# In[9]:


print([0.5 for _ in range(CHANNELS_IMG)])


# In[10]:


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)])
    ]
)


# In[11]:


dataset = datasets.MNIST(root='./dataset/',train=True,transform=transforms,download=True)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
initalize_weights(gen)
initalize_weights(disc)
opt_gen = Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
opt_disc = Adam(disc.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
criterion = BCELoss()
fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)


# In[12]:


gen.train()
disc.train()


# In[13]:


wandb.init(project=PROJECT_NAME,name='baseline')
for _ in tqdm(range(EPOCHS)):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_dics = (loss_disc_fake + loss_disc_real) / 2
        disc.zero_grad()
        loss_dics.backward(retain_graph=True)
        opt_disc.step()
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        wandb.log({'Loss Disc':loss_dics.item()})
        wandb.log({'Loss Gen':loss_gen.item()})
        img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
        wandb.log({'Img Fake':wandb.Image(img_grid_fake)})
        wandb.log({'Img Real':wandb.Image(img_grid_real)})
wandb.finish()


# In[ ]:




