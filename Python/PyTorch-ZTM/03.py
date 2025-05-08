#!/usr/bin/env python
# coding: utf-8

# # PyTorch Computer Vision

# ## Computer vision libraries in PyTorch
# 
# * [`torchvision`](https://pytorch.org/vision/stable/index.html) - base domain library of PyTorch Computer Vision
# * `torchvision.datasets` - get datasets and data loading functions for computer vision here
# * `torchvison.models` - get pretrained computer vision models that you can leverage for you own problems
# * `torchvison.transforms` - functions for manipulating your vision data (images) to be suitable for use with an ML model
# * `torch.utils.data.Dataset` - Base dataset class for PyTorch
# * `torch.utils.data.DataLoader` - Creaters a Python iterable over a dataset

# In[1]:


import torch
import torchvision
from torch import nn, optim
from torchvision import datasets,models,transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


torch.__version__,torchvision.__version__


# ## Dataset
# 
# The dataset we'll be using is FashionMNIST

# In[3]:


# Setup training data
train_data = datasets.FashionMNIST(root="data/03/",train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="data/03/",train=False,download=True,transform=transforms.ToTensor())


# In[4]:


len(train_data),len(test_data)


# In[5]:


# See the first training example
image,label = train_data[0]


# In[6]:


image.shape,label


# In[7]:


class_names = train_data.classes


# In[8]:


class_names


# In[9]:


class_to_idx = train_data.class_to_idx


# In[10]:


class_to_idx


# In[11]:


image.shape,label # C,H,W


# In[12]:


plt.imshow(image.view(28,28,1),cmap="gray")
plt.title(class_names[label])
plt.axis(False);


# In[13]:


# Plot more images
# torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
row,cols = 4,4
for i in range(1,row*cols+1):
    random_idx = torch.randint(0,len(train_data),size=[1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(row,cols,i)
    plt.imshow(img.view(28,28,1),cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)


# ## PrePare DataLoader
# 
# Currently the data is inform of PyTorch Datasets
# 
# DataLoader turns our data to a Python iterable
# 
# We will turn out data into mini batches
# 
# Why?
# - Computationally efficent
# - Neural network can change the gradient more times, it can change the weights and biases every batch rather than only once per epoch

# In[14]:


BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)


# In[15]:


train_features_batch,train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape,train_labels_batch.shape


# In[16]:


rdm_idx = torch.randint(0,len(train_features_batch),size = [1]).item()
img,label = train_features_batch[rdm_idx],train_labels_batch[rdm_idx]
plt.imshow(img.squeeze(),cmap="gray")
plt.title(class_names[label])
plt.axis(False)


# ## Model 0: Build a baseline model
# 
# When starting to build a series of machine learning modelling experiement, its best to start with a baseline model

# In[17]:


# Create a flatten layer
flatten_model = nn.Flatten()
x = train_features_batch[0]
x.shape,flatten_model(x).shape


# In[18]:


from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self,input_shape:int,hidden_units: int, output_shape:int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # No learning parameters
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )
    
    def forward(self,X) -> torch.Tensor():
        return self.layer_stack(X)


# In[19]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[20]:


torch.manual_seed(42)

# Setup model with input parameters

model_0 = FashionMNISTModelV0(input_shape=28*28,hidden_units=10,output_shape=len(class_names))


# In[21]:


optimizer = optim.Adam(model_0.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()


# In[22]:


dummy_X = torch.rand([1,1,28,28])
model_0(dummy_X)


# ### Setup loss, optimizer and evaluation metrics
# 
# * Loss function - since we're working with multi class data, our loss function with be `nn.CrossEntropyLoss()`
# * Optimizer - our optimizer is gonna be SGD but I am gonna use Adam() ;)
# * For evaluation we are gonan be using accuracy

# In[23]:


from helper_functions import *


# In[24]:


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_0.parameters(),lr=0.01)


# ### Creating a function to time our experiments
# 
# Machine learning is very experimental
# 
# 2 of the main things we would want to track are:
# 1. Model's performence
# 2. How fast the model runs

# In[25]:


from timeit import default_timer as timer


# In[26]:


def print_train_time(start:float, end:float, device:torch.device = None):
    total_time = end - start
    print(f"Process took - {total_time}")


# In[27]:


start_time = timer()
# some code..
end_time = timer()
print_train_time(start=start_time,end=end_time)


# ### Creating a training loop and training a model on batches of data
# 
# 1. Loop through epochs
# 2. Loop through training batches,perform training steps, calculate the train loss per batch
# 3. Loop through testing batches, perform testing steps, calculate test loss per batch
# 4. Print out what's happening
# 5. Time it all

# In[28]:


from tqdm.auto import tqdm


# In[29]:


epochs = 3


# In[30]:


# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer() 

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n------")
  ### Training
  train_loss = 0
  # Add a loop to loop through the training batches
  for batch, (X, y) in enumerate(train_dataloader):
    model_0.train()
    # 1. Forward pass
    y_pred = model_0(X)
    
    # 2. Calculate loss (per batch)
    loss = loss_fn(y_pred, y)
    train_loss += loss # accumulate train loss
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4. Loss backward
    loss.backward()
    
    # 5. Optimizer step (update the model's parameters once *per batch*)
    optimizer.step()
    
    # Print out what's happening
    if batch % 400 == 0:
        print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
  
  # Divide total train loss by length of train dataloader
  train_loss /= len(train_dataloader)

  ### Testing
  test_loss, test_acc = 0, 0
  model_0.eval()
  with torch.inference_mode(): 
    for X_test, y_test in test_dataloader:
      # 1. Forward pass
      test_pred = model_0(X_test)

      # 2. Calculate loss (accumulatively)
      test_loss += loss_fn(test_pred, y_test)

      # 3. Calculate accuracy
      test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

    # Calculate the test loss average per batch
    test_loss /= len(test_dataloader)

    # Calculate the test acc average per batch
    test_acc /= len(test_dataloader)

  # Print out what's happening
  print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))


# ## Make predicitions and get Model 0 results

# In[31]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[32]:


def eval_model(model:torch.nn.Module, 
               data_loader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module,
               accuracy_fn,
               train_or_loss:str):
    loss,acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            preds = model(X.to(device))
            f_preds = torch.argmax(preds,dim=1)
            loss += loss_fn(preds.to(device),y.to(device))
            acc += accuracy_fn(y_true=y.to(device),y_pred=f_preds.to(device))
        # Scale loss and acc to find average loss/acc per batch
        loss = loss / len(data_loader)
        acc = acc / len(data_loader)
    return {f"{train_or_loss} model_loss":loss.item(),f"{train_or_loss} model_acc":acc,"model_name":model.__class__.__name__,}


# In[33]:


model_0_results = eval_model(model=model_0.to(device),data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="test")


# In[34]:


# eval_model(model=model_0,data_loader=train_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="train")


# ## Setup device agnostic code

# In[35]:


get_ipython().system('nvidia-smi')


# In[36]:


# Batchnorm is before activation
class FashionMNIST01(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28,512)
        self.batchnorm1d = nn.BatchNorm1d(512,affine=True)
        self.linear2 = nn.Linear(512,1024)
        self.batchnorm2d = nn.BatchNorm1d(1024,affine=True)
        self.linear3 = nn.Linear(1024,2048)
        self.batchnorm3d = nn.BatchNorm1d(2048,affine=True)
        self.linear4 = nn.Linear(2048,1024)
        self.batchnorm4d = nn.BatchNorm1d(1024,affine=True)
        self.output = nn.Linear(1024,len(class_names))
    
    def forward(self,X):
        X = self.flatten(X)
        X = self.activation(self.batchnorm1d(self.linear1(X)))
        X = self.activation(self.batchnorm2d(self.linear2(X)))
        X = self.activation(self.batchnorm3d(self.linear3(X)))
        X = self.activation(self.batchnorm4d(self.linear4(X)))
        X = self.output(X)
        return X


# In[37]:


model_1 = FashionMNIST01().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(),lr=0.01)
epochs = 1


# In[38]:


import wandb


# In[39]:


wandb.init(project="03",name="model_1_affine_on")
for epoch in tqdm(range(epochs)):
    for X_batch,y_batch in train_dataloader:
        torch.cuda.empty_cache()
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_logits = model_1(X_batch)
        preds = torch.argmax(y_logits,dim=1)
        loss = criterion(y_logits,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # wandb.log(eval_model(model=model_1,data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="test"))
    # wandb.log(eval_model(model=model_1,data_loader=train_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="train"))
wandb.finish()


# In[40]:


def train_step(dataloader:torch.utils.data.DataLoader,model:nn.Module,criterion:torch.nn,optimizer:torch.optim):
    for X_batch,y_batch in dataloader:
        torch.cuda.empty_cache()
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_logits = model(X_batch)
        preds = torch.argmax(y_logits,dim=1)
        loss = criterion(y_logits,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


# In[41]:


def test_step(dataloader:torch.utils.data.DataLoader,model:nn.Module,accuracy_fn:accuracy_fn):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode(): 
        for X_test, y_test in dataloader:
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss,test_acc


# In[42]:


model_1_results = eval_model(model=model_1,data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="test")


# In[43]:


eval_model(model=model_1,data_loader=train_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="train")


# ## Model 2: Building a convolutioanl neural network
# 
# CNN's are also know as CovNet
# 
# Thry are infamous for their great performence with images

# In[44]:


# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
  """
  Model architecture that replicates the TinyVGG
  model from CNN explainer website.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        # Create a conv layer - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        nn.Conv2d(in_channels=input_shape, 
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1), # values we can set ourselves in our NN's are called hyperparameters
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, # there's a trick to calculating this...
                  out_features=256),
        nn.ReLU(),
        nn.Linear(256,512),
        nn.ReLU(),
        nn.Linear(512,1024),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    # print(f"Output shape of conv_block_1: {x.shape}")
    x = self.conv_block_2(x) 
    # print(f"Output shape of conv_block_2: {x.shape}")
    x = self.classifier(x)
    # print(f"Output shape of classifier: {x.shape}")
    return x


# In[45]:


# model = FashionMNISTModelV2(input_shape=1,
#                               hidden_units=10,
#                               output_shape=len(class_names)).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model.parameters(),
#                             lr=0.1)
# epochs = 100


# ### Stepping through `nn.Conv2d`

# In[46]:


torch.manual_seed(42)
images = torch.randn(size=(32,3,64,64))
test_image = images[0]


# In[47]:


test_image.shape


# In[48]:


cov_layer = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=(3,3),stride=1,padding=0)


# In[49]:


cov_layer(test_image.unsqueeze(0)).shape,test_image.unsqueeze(0).shape


# ### Stepping through `nn.MaxPool2d()`

# In[50]:


test_image.shape


# In[51]:


plt.imshow(test_image.view(64,64,3))


# In[52]:


# Create sample nn.MaxPool2d layer

max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass through conv layer
test_image_through_conv = cov_layer(test_image.unsqueeze(dim=0))
print(test_image_through_conv.shape)

# Pass through max pool 2d
test_image_through_maxpool2d = max_pool_layer(test_image_through_conv)
print(test_image_through_maxpool2d.shape)


# In[53]:


torch.manual_seed(42)
random_tensor =torch.randn(size=(1,1,2,2))


# In[54]:


nn.MaxPool2d(kernel_size=2)(random_tensor)


# In[55]:


rand_image_tensor = torch.randn(size=(1,28,28))
rand_image_tensor.shape


# In[56]:


# model(rand_image_tensor.unsqueeze(0).to(device))


# In[57]:


model = FashionMNISTModelV2(input_shape=1,
                              hidden_units=256,
                              output_shape=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)
epochs = 15


# In[58]:


wandb.init(project="03",name="CNN")
for epoch in tqdm(range(epochs)):
    for X_batch,y_batch in train_dataloader:
        torch.cuda.empty_cache()
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_logits = model(X_batch)
        preds = torch.argmax(y_logits,dim=1)
        loss = criterion(y_logits,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        total_loss = 0
        total_accuracy = 0
        for X_test,y_test in test_dataloader:
            preds = model(X_test.to(device))
            loss = criterion(preds.to(device),y_test.to(device))
            total_loss += loss
            preds = torch.argmax(preds,dim=1)
            total_accuracy += accuracy_fn(y_true=y_test.to(device), y_pred=preds.to(device))
        accuracy_test = total_accuracy / (len(test_data)/32)
        loss_test = total_loss / (len(test_data)/32)
        total_loss = 0
        total_accuracy = 0
        for X_train,y_train in train_dataloader:
            preds = model(X_train.to(device))
            loss = criterion(preds.to(device),y_train.to(device))
            total_loss += loss
            preds = torch.argmax(preds,dim=1)
            total_accuracy += accuracy_fn(y_true=y_train.to(device), y_pred=preds.to(device))
        accuracy_train = total_accuracy / (len(train_data)/32)
        loss_train = total_loss / (len(train_data)/32)
    wandb.log({
        "train model_loss":loss_train,
        "test model_acc":accuracy_test,
        "test model_loss":loss_test,
        "train model_acc":accuracy_train
    })
wandb.finish()


# In[59]:


model_2_results = eval_model(model=model_1,data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,train_or_loss="test")
import pandas as pd
compare_results = pd.DataFrame([model_0_results,model_1_results,model_2_results])
compare_results


# In[60]:


compare_results.set_index("test model_loss")


# ## Make and Evaluate random predictions with best model

# In[61]:


def make_predictions(model,data,device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample,dim=0).to(device)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze(),dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)


# In[88]:


import random
test_samples = []
test_labels = []
for sample,label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)


# In[89]:


plt.imshow(test_samples[0].squeeze(),cmap="gray")
plt.title(class_names[test_labels[0]])


# In[90]:


pred_probs = make_predictions(model=model,data=test_samples,device=device)
pred_clases = pred_probs.argmax(dim=1)
plt.figure(figsize=(9,9))
nrows=3
ncols = 3
for i,sample in enumerate(test_samples):
    plt.subplot(nrows,ncols,i+1)
    plt.imshow(sample.squeeze(),cmap="gray")
    pred_label = class_names[pred_clases[i]]
    truth_labe = class_names[test_labels[i]]
    title_text = f"Correct : {truth_labe} | Model : {pred_label}"
    if pred_label == truth_labe:
        plt.title(title_text,fontsize=10,c='g')
    else:
        plt.title(title_text,fontsize=10,c='r')
    plt.axis(False);


# ## making a confusion matrix for further prediction evaluation
# 
# 1. Make predictions with our trained model on the test dataset
# 2. Make a confusion matrix `torchmetrics.confusionmatrix()`
# 3. plot the confusion matrix with `mlxten`

# In[92]:


from tqdm import tqdm

y_preds = []
model.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader,desc="Making predictions..."):
        X,y = X.to(device),y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit.squeeze(),dim=0).argmax(dim=1)
        y_preds.append(y_pred.to('cpu'))
y_pred_tensor = torch.cat(y_preds)
y_pred_tensor[:10]


# In[105]:


import torchmetrics,mlxtend


# In[106]:


mlxtend.__version__


# In[107]:


torchmetrics.__version__


# In[108]:


get_ipython().system('pip install mlxtend==0.19.0')


# In[109]:


from torchmetrics  import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
confmat = ConfusionMatrix(num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor,target=test_data.targets)
fig,ax = plot_confusion_matrix(conf_mat=confmat_tensor.numoy(),class_names=class_names,figsize=(10,7))


# In[ ]:




