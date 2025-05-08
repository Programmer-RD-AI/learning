#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy.lib import stride_tricks
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os


class Modelling:
    def train(
        self,
        model,
        X,
        optimizer,conv2d_output,
        y,
        criterion,
        EPOCHS=250,
        BATCH_SIZE=32,
        PROJECT_NAME="test",
        NAME="test",
        device=torch.device("cuda"),
        IMG_SIZE=224,
    ):
        model.to(device)
        lossess = []
        wandb.init(project=PROJECT_NAME, name=NAME)
        for epoch in tqdm(range(EPOCHS)):
            for i in range(0, len(X), BATCH_SIZE):
                X_batch = X[i : i + BATCH_SIZE]
                y_batch = y[i : i + BATCH_SIZE]
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch.view(-1, 1, IMG_SIZE, IMG_SIZE).float())
                preds.to(device)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lossess.append(loss.item())
                wandb.log(
                    {
                        "loss_per_batch": loss.item(),
                    }
                )
            wandb.log(
                    {
                        "loss": loss.item(),
                    }
            )
        return preds,y_batch


class Model(nn.Module):
    def __init__(self,conv2d_output=64,conv2d_1_ouput=32,output_fc1=128,max_pool2d=2,num_of_linear=1,activation=nn.ReLU()):
        super().__init__()
        print(conv2d_output)
        print(conv2d_1_ouput)
        print(output_fc1)
        print(max_pool2d)
        print(num_of_linear)
        print(activation)
        self.conv2d_output = conv2d_output
        self.conv2d_1_ouput = conv2d_1_ouput
        self.conv1 = nn.Conv2d(1, conv2d_1_ouput,3)
        self.conv3 = nn.Conv2d(conv2d_1_ouput,conv2d_1_ouput,3)
        self.conv2 = nn.Conv2d(conv2d_1_ouput, conv2d_output, 3)
        self.fc1 = nn.Linear(
            self.conv2d_output*26*26, output_fc1
        )  # output_channel*max_pool2d_kernal*max_pool2d_kernal
        self.fc2 = nn.Linear(output_fc1, output_fc1)
        self.fc3 = nn.Linear(output_fc1, 2)
        self.relu = activation
        self.max_pool2d = nn.MaxPool2d(max_pool2d)
        self.num_of_linear = num_of_linear

    def forward(self, X):
        preds = self.conv1(X)
        preds = self.relu(preds)
        preds = self.max_pool2d(preds)
        preds = self.conv3(preds)
        preds = self.relu(preds)
        preds = self.max_pool2d(preds)
        preds = self.conv2(preds)
        preds = self.relu(preds)
        preds = self.max_pool2d(preds)
        preds = preds.view(-1, self.conv2d_output*26*26)
        preds = self.fc1(preds)
        preds = self.relu(preds)
        for _ in range(self.num_of_linear):
            preds = self.fc2(preds)
            preds = self.relu(preds)
        preds = self.fc3(preds)
        preds = F.softmax(preds, dim=1)
        return preds


# In[2]:


import numpy as np
import cv2
import os
from tqdm import tqdm


class Load_Data:
    def __init__(self, IMG_SIZE=224):
        self.IMG_SIZE = IMG_SIZE
        self.data = []
        self.LABELS = {"./data/raw/Mask/": [0, -1], "./data/raw/Non Mask/": [1, -1]}

    def load_data(self, IMREAD_TYPE=cv2.IMREAD_GRAYSCALE):
        for label in tqdm(self.LABELS):
            for filename in os.listdir(label):
                try:
                    filepath = label + filename
                    img = cv2.imread(filepath, IMREAD_TYPE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    img = img / 255.0
                    self.data.append([np.array(img), np.eye(2)[self.LABELS[label][0]]])
                    self.LABELS[label][1] = self.LABELS[label][1] + 1
                except Exception as e:
                    print(e)
                    # pass
        for l in self.LABELS:
            print(f"Class : {l} \n Count : {self.LABELS[l][1]}")
        np.random.shuffle(self.data)
        np.save("./data/cleaned/data.npy", self.data)
        return self.data


# ## Load the data

# In[3]:


def load_data():
    ld = Load_Data()
    data = ld.load_data()
    X = []
    y = []
    for d in data:
        X.append(d[0])
        y.append(d[1])
    VAL_SPLIT = 0.25
    VAL_SPLIT = len(X)*VAL_SPLIT
    VAL_SPLIT = int(VAL_SPLIT)
    X_train = X[:-VAL_SPLIT]
    y_train = y[:-VAL_SPLIT]
    X_test = X[-VAL_SPLIT:]
    y_test = y[-VAL_SPLIT:]
    X_train = torch.from_numpy(np.array(X_train).astype(np.float32))
    y_train = torch.from_numpy(np.array(y_train).astype(np.float32))
    X_test = torch.from_numpy(np.array(X_test).astype(np.float32))
    y_test = torch.from_numpy(np.array(y_test).astype(np.float32))
    return [[X,y],[[X_train,y_train],[X_test,y_test]]]


# In[4]:


data = load_data()


# In[5]:


import numpy as np
# np.random.shuffle(data)


# In[6]:


X,y = data[0]


# In[7]:


X = np.array(X)
y = np.array(y)


# In[8]:


X_train,y_train = data[1][0]


# In[9]:


X_test,y_test = data[1][1]


# In[10]:


X_train.shape


# In[11]:


y_train.shape


# In[12]:


X_test.shape


# In[13]:


y_test.shape


# In[14]:


X.shape


# In[15]:


y.shape


# ## Modelling

# In[16]:


modelling = Modelling()


# In[17]:


model = Model()
loss_funtion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)


# In[18]:


PROJECT_NAME = 'Mask-or-Not'


# In[19]:


device = torch.device('cuda')


# In[20]:


model = model.to('cuda')


# In[21]:


X_train = X_train.to(device)
y_train = y_train.to(device)


# In[22]:


# preds,y_real = modelling.train(model,X_train,optimizer,y_train,loss_funtion,PROJECT_NAME=PROJECT_NAME,NAME='testing-0',device=torch.device('cuda'))


# In[23]:


loss_logs = []


# In[24]:


IMG_SIZE = 224


# In[25]:


model = Model().to(device)
loss_funtion = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)


# In[26]:


device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


# In[27]:


device


# In[28]:


# preds,y_true = modelling.train(model,X_train,optimizer,y_train,loss_funtion,PROJECT_NAME=PROJECT_NAME,NAME='L1Loss',device=torch.device('cuda'),EPOCHS=1000)


# In[29]:


def test(model,X_train,y_train):
    total = 0
    correct = 0
    for i in range(len(X_train)):
        input_img = X_train[i].view(-1,1,224,224).to('cpu')
        model.to('cpu')
        pred = model(input_img).detach().numpy()
        pred = torch.from_numpy(pred)
        pred = torch.argmax(pred)
        target = y_train[i]
        target = torch.argmax(target)
        if target == pred:
            correct += 1
        input_img = X_train[i-1].view(-1,1,224,224).to('cpu')
        pred_old = model(input_img).detach().numpy()
        pred_old = torch.argmax(torch.from_numpy(pred_old))
        total += 1
    return correct,total


# In[30]:


# total = 0
# correct = 0
# same = 0
# for i in range(len(X_train)):
#     input_img = X_train[i].view(-1,1,224,224).to('cpu')
#     model.to('cpu')
#     pred = model(input_img).detach().numpy()
#     pred = torch.from_numpy(pred)
#     pred = torch.argmax(pred)
#     target = y_train[i]
#     target = torch.argmax(target)
#     if target == pred:
#         correct += 1
#     input_img = X_train[i-1].view(-1,1,224,224).to('cpu')
#     pred_old = model(input_img).detach().numpy()
#     pred_old = torch.argmax(torch.from_numpy(pred_old))
#     if str(pred) != str(pred_old):
#         print(f'!=0 Now : {pred}/{y_train[i]}')
#         print(f'!=0 Old : {pred_old}/{y_train[i-1]}')
#         print('\n')
#         same += 1
#     total += 1


# In[31]:


# BCELoss
# 0.498 25 Loss
# MSELoss
# 0.497 0.25 Loss
# L1Loss
# 0.497 0.25 not best as MSELoss


# In[32]:


# round(correct/total,3)


# ## Testing modelling

# In[33]:


device = torch.device('cuda')


# In[34]:


BATCH_SIZE = 128


# In[35]:


# EPOCHS = 12
# wandb.init(project=PROJECT_NAME, name='test')
# for epoch in tqdm(range(EPOCHS)):
#     for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
#         X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
#         y_batch = y_train[i : i + BATCH_SIZE]
#         X_batch = X_batch.to('cuda')
#         y_batch = y_batch.to('cuda')
#         X_batch = X_batch.cuda()
#         model.to('cuda')
#         preds = model(X=X_batch)
#         preds.to(device)
#         loss = loss_funtion(preds, y_batch)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         wandb.log({
#         'loss_iter':loss.item(),
#         'accuracy_iter':round(test(model,X_train,y_train)[0],test(model,X_train,y_train)[1]),
#         'val_accuracy_iter':round(test(model,X_test,y_test)[0],test(model,X_test,y_test)[1])
#         })
#     wandb.log({
#         'loss':loss.item(),
#         'accuracy':round(test(model,X_train,y_train)[0],test(model,X_train,y_train)[1]),
#         'val_accuracy':round(test(model,X_test,y_test)[0],test(model,X_test,y_test)[1])
#     })


# In[36]:


BATCH_SIZE = 32


# In[37]:


# conv2d_output
# conv2d_1_ouput
# output_fc1
# max_pool2d
# num_of_linear
# activation
# best num of epochs
# best optimizer
# best loss
## best lr


# In[38]:


# best num of epochs
## best lr


# In[39]:


BATCH_SIZE = 250


# In[ ]:


# best optimizer
optimizers = [torch.optim.Adam,torch.optim.AdamW,torch.optim.Adamax,torch.optim.SGD]
for optimizer in optimizers:
    EPOCHS = 3
    model = Model().to('cuda')
    loss_funtion = torch.nn.MSELoss().to('cuda')
    optimizer = optimizer(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'optimizer-{optimizer}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


# num_of_linears
activations = [nn.ELU(),nn.LeakyReLU(),nn.PReLU(),nn.ReLU(),nn.ReLU6(),nn.RReLU(),nn.SELU(),nn.CELU(),nn.GELU(),nn.SiLU(),nn.Tanh()]
for activation in activations:
    EPOCHS = 3
    model = Model(activation=activation).to('cuda')
    loss_funtion = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'activation-{activation}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


# num_of_linears
num_of_linears = [1,2,3,4,5]
for num_of_linear in num_of_linears:
    EPOCHS = 3
    model = Model(num_of_linear=num_of_linear).to('cuda')
    loss_funtion = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'num_of_linear-{num_of_linear}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


# max_pool2d
max_pool2ds = [1,2,5,7,10]
for max_pool2d in max_pool2ds:
    EPOCHS = 3
    model = Model(max_pool2d=max_pool2d).to('cuda')
    loss_funtion = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'max_pool2d-{max_pool2d}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


# output_fc1
output_fc1s = [16,32,64,128]
for output_fc1 in output_fc1:
    EPOCHS = 3
    model = Model(output_fc1=output_fc1).to('cuda')
    loss_funtion = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'output_fc1-{output_fc1}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


# conv2d_1_ouput
conv2d_1_ouputs = [16,32,64,128]
for conv2d_1_ouput in conv2d_1_ouputs:
    EPOCHS = 3
    model = Model(conv2d_output=256,conv2d_1_ouput=conv2d_1_ouput).to('cuda')
    loss_funtion = torch.nn.MSELoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'conv2d_1_ouputs-{conv2d_1_ouput}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


# # conv2d_1_ouput
# conv2d_1_ouputs = [16,32,64,128,256]
# for conv2d_1_ouput in conv2d_1_ouputs:
#     EPOCHS = 3
#     model = Model(conv2d_output=256,conv2d_1_ouput=conv2d_1_ouput).to('cuda')
#     loss_funtion = torch.nn.MSELoss().to('cuda')
#     optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
#     wandb.init(project=PROJECT_NAME, name=f'conv2d_1_ouputs-{conv2d_1_ouput}')
#     for epoch in tqdm(range(EPOCHS)):
#         for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
#             X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
#             y_batch = y_train[i : i + BATCH_SIZE]
#             X_batch = X_batch.to('cuda')
#             y_batch = y_batch.to('cuda')
#             X_batch = X_batch.cuda()
#             model.to('cuda')
#             preds = model(X=X_batch)
#             preds.to(device)
#             loss = loss_funtion(preds, y_batch)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             print({
#                 'loss':loss.item(),
#                 'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
#                 'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
#             })
#             wandb.log({
#                 'loss':loss.item(),
#                 'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
#                 'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
#             })


# In[ ]:


# best loss
lossess = [nn.L1Loss,nn.MSELoss,torch.nn.HingeEmbeddingLoss,torch.nn.MarginRankingLoss,torch.nn.TripletMarginLoss]
for loss_funtion in lossess:
    EPOCHS = 3
    model = Model().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.1)
    wandb.init(project=PROJECT_NAME, name=f'loss_funtion-{loss_funtion}')
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
            X_batch = X_train[i : i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            y_batch = y_train[i : i + BATCH_SIZE]
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            X_batch = X_batch.cuda()
            model.to('cuda')
            preds = model(X=X_batch)
            preds.to(device)
            loss = loss_funtion(preds, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })
            wandb.log({
                'loss':loss.item(),
                'accuracy':test(model,X_train,y_train)[0]/test(model,X_train,y_train)[1],
                'val_accuracy':test(model,X_test,y_test)[0]/test(model,X_test,y_test)[1]
            })


# In[ ]:


BATCH_SIZE = 32


# In[ ]:


# conv2d_output
# conv2d_1_ouput
# output_fc1
# max_pool2d
# num_of_linear
# activation
# best num of epochs
# best optimizer
# best loss
## best lr


# In[ ]:




