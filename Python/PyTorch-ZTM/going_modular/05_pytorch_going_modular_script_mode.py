#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 05. Going Modular: Part 2 (script mode)
# 
# This notebook is part 2/2 of section [05. Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/).
# 
# For reference, the two parts are: 
# 1. [**05. Going Modular: Part 1 (cell mode)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb) - this notebook is run as a traditional Jupyter Notebook/Google Colab notebook and is a condensed version of [notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/).
# 2. [**05. Going Modular: Part 2 (script mode)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb) - this notebook is the same as number 1 but with added functionality to turn each of the major sections into Python scripts, such as, `data_setup.py` and `train.py`. 
# 
# Why two parts?
# 
# Because sometimes the best way to learn something is to see how it *differs* from something else.
# 
# If you run each notebook side-by-side you'll see how they differ and that's where the key learnings are.

# ## What is script mode?
# 
# **Script mode** uses [Jupyter Notebook cell magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) (special commands) to turn specific cells into Python scripts.
# 
# For example if you run the following code in a cell, you'll create a Python file called `hello_world.py`:
# 
# ```
# %%writefile hello_world.py
# print("hello world, machine learning is fun!")
# ```
# 
# You could then run this Python file on the command line with:
# 
# ```
# python hello_world.py
# 
# >>> hello world, machine learning is fun!
# ```
# 
# The main cell magic we're interested in using is `%%writefile`.
# 
# Putting `%%writefile filename` at the top of a cell in Jupyter or Google Colab will write the contents of that cell to a specified `filename`.
# 
# > **Question:** Do I have to create Python files like this? Can't I just start directly with a Python file and skip using a Google Colab notebook?
# >
# > **Answer:** Yes. This is only *one* way of creating Python scripts. If you know the kind of script you'd like to write, you could start writing it straight away. But since using Jupyter/Google Colab notebooks is a popular way of starting off data science and machine learning projects, knowing about the `%%writefile` magic command is a handy tip. 

# ## What has script mode got to do with PyTorch?
# 
# If you've written some useful code in a Jupyter Notebook or Google Colab notebook, chances are you'll want to use that code again.
# 
# And turning your useful cells into Python scripts (`.py` files) means you can use specific pieces of your code in other projects.
# 
# This practice is not PyTorch specific.
# 
# But it's how you'll see many different online PyTorch repositories structured.

# ### PyTorch in the wild
# 
# For example, if you find a PyTorch project on GitHub, it may be structured in the following way:
# 
# ```
# pytorch_project/
# ├── pytorch_project/
# │   ├── data_setup.py
# │   ├── engine.py
# │   ├── model.py
# │   ├── train.py
# │   └── utils.py
# ├── models/
# │   ├── model_1.pth
# │   └── model_2.pth
# └── data/
#     ├── data_folder_1/
#     └── data_folder_2/
# ```
# 
# Here, the top level directory is called `pytorch_project` but you could call it whatever you want.
# 
# Inside there's another directory called `pytorch_project` which contains several `.py` files, the purposes of these may be:
# * `data_setup.py` - a file to prepare data (and download data if needed).
# * `engine.py` - a file containing various training functions.
# * `model_builder.py` or `model.py` - a file to create a PyTorch model.
# * `train.py` - a file to leverage all other files and train a target PyTorch model.
# * `utils.py` - a file dedicated to helpful utility functions.
# 
# And the `models` and `data` directories could hold PyTorch models and data files respectively (though due to the size of models and data files, it's unlikely you'll find the *full* versions of these on GitHub, these directories are present above mainly for demonstration purposes).
# 
# > **Note:** There are many different ways to structure a Python project and subsequently a PyTorch project. This isn't a guide on *how* to structure your projects, only an example of how you *might* come across PyTorch projects in the wild. For more on structuring Python projects, see Real Python's [*Python Application Layouts: A Reference*](https://realpython.com/python-application-layouts/) guide.

# ## What's the difference between this notebook (Part 2) and the cell mode notebook (Part 1)?
# 
# This notebook, 05 Going Modular: Part 2 (script mode), creates Python scripts out of the cells created in part 1.
# 
# Running this notebook end-to-end will result in having a directory structure very similar to the `pytorch_project` structure above.
# 
# You'll notice each section in Part 2 (script mode) has an extra subsection (e.g. 2.1, 3.1, 4.1) for turning cell code into script code.

# ## What we're going to cover
# 
# By the end of this notebook you should finish with a directory structure of: 
# 
# ```
# going_modular/
# ├── going_modular/
# │   ├── data_setup.py
# │   ├── engine.py
# │   ├── model_builder.py
# │   ├── train.py
# │   └── utils.py
# ├── models/
# │   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
# │   └── 05_going_modular_script_mode_tinyvgg_model.pth
# └── data/
#     └── pizza_steak_sushi/
#         ├── train/
#         │   ├── pizza/
#         │   │   ├── image01.jpeg
#         │   │   └── ...
#         │   ├── steak/
#         │   └── sushi/
#         └── test/
#             ├── pizza/
#             ├── steak/
#             └── sushi/
# ```
# 
# Using this directory structure, you should be able to train a model from within a notebook with the command:
# 
# ```
# !python going_modular/train.py
# ```
# 
# Or from the command line with:
# 
# ```
# python going_modular/train.py
# ```
# 
# In essence, we will have turned our helpful notebook code into **reusable modular code**.

# ## Where can you get help?
# 
# You can find the book version of this section [05. PyTorch Going Modular on learnpytorch.io](https://www.learnpytorch.io/05_pytorch_going_modular/).
# 
# The rest of the materials for this course [are available on GitHub](https://github.com/mrdbourke/pytorch-deep-learning).
# 
# If you run into trouble, you can ask a question on the course [GitHub Discussions page](https://github.com/mrdbourke/pytorch-deep-learning/discussions).
# 
# And of course, there's the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) and [PyTorch developer forums](https://discuss.pytorch.org/), a very helpful place for all things PyTorch. 

# ## 0. Creating a folder for storing Python scripts
# 
# Since we're going to be creating Python scripts out of our most useful code cells, let's create a folder for storing those scripts.
# 
# We'll call the folder `going_modular` and create it using Python's [`os.makedirs()`](https://docs.python.org/3/library/os.html) method.

# In[1]:


import os

os.makedirs("going_modular", exist_ok=True)


# ## 1. Get data
# 
# We're going to start by downloading the same data we used in [notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#1-get-data), the `pizza_steak_sushi` dataset with images of pizza, steak and sushi.

# In[2]:


import os
import zipfile

from pathlib import Path

import requests

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")


# In[3]:


# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir


# ## 2. Create Datasets and DataLoaders
# 
# Let's turn our data into PyTorch `Dataset`'s and `DataLoader`'s and find out a few useful attributes from them such as `classes` and their lengths. 

# In[4]:


from torchvision import datasets, transforms

# Create simple transform
data_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")


# In[5]:


# Get class names as a list
class_names = train_data.classes
class_names


# In[6]:


# Can also get class names as a dict
class_dict = train_data.class_to_idx
class_dict


# In[7]:


# Check the lengths
len(train_data), len(test_data)


# In[8]:


# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

train_dataloader, test_dataloader


# In[9]:


# Check out single image size/shape
img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")


# ### 2.1 Create Datasets and DataLoaders (script mode)
# 
# Rather than rewriting all of the code above everytime we wanted to load data, we can turn it into a script called `data_setup.py`.
# 
# Let's capture all of the above functionality into a function called `create_dataloaders()`.

# In[10]:


get_ipython().run_cell_magic('writefile', 'going_modular/data_setup.py', '"""\nContains functionality for creating PyTorch DataLoaders for \nimage classification data.\n"""\nimport os\n\nfrom torch.utils.data import DataLoader\nfrom torchvision import datasets, transforms\n\nNUM_WORKERS = os.cpu_count()\n\ndef create_dataloaders(\n    train_dir: str, \n    test_dir: str, \n    transform: transforms.Compose, \n    batch_size: int, \n    num_workers: int=NUM_WORKERS\n):\n  """Creates training and testing DataLoaders.\n\n  Takes in a training directory and testing directory path and turns\n  them into PyTorch Datasets and then into PyTorch DataLoaders.\n\n  Args:\n    train_dir: Path to training directory.\n    test_dir: Path to testing directory.\n    transform: torchvision transforms to perform on training and testing data.\n    batch_size: Number of samples per batch in each of the DataLoaders.\n    num_workers: An integer for number of workers per DataLoader.\n\n  Returns:\n    A tuple of (train_dataloader, test_dataloader, class_names).\n    Where class_names is a list of the target classes.\n    Example usage:\n      train_dataloader, test_dataloader, class_names = \\\n        = create_dataloaders(train_dir=path/to/train_dir,\n                             test_dir=path/to/test_dir,\n                             transform=some_transform,\n                             batch_size=32,\n                             num_workers=4)\n  """\n  # Use ImageFolder to create dataset(s)\n  train_data = datasets.ImageFolder(train_dir, transform=transform)\n  test_data = datasets.ImageFolder(test_dir, transform=transform)\n\n  # Get class names\n  class_names = train_data.classes\n\n  # Turn images into data loaders\n  train_dataloader = DataLoader(\n      train_data,\n      batch_size=batch_size,\n      shuffle=True,\n      num_workers=num_workers,\n      pin_memory=True,\n  )\n  test_dataloader = DataLoader(\n      test_data,\n      batch_size=batch_size,\n      shuffle=False,\n      num_workers=num_workers,\n      pin_memory=True,\n  )\n\n  return train_dataloader, test_dataloader, class_names\n')


# ## 3. Making a model (TinyVGG) 
# 
# We're going to use the same model we used in notebook 04: TinyVGG from the CNN Explainer website.
# 
# The only change here from notebook 04 is that a docstring has been added using [Google's Style Guide for Python](https://google.github.io/styleguide/pyguide.html#384-classes). 

# In[11]:


import torch

from torch import nn 

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion


# Now let's create an instance of `TinyVGG` and put it on the target device.
# 
# > **Note:** If you're using Google Colab, and you'd like to use a GPU (recommended), you can turn one on via going to Runtime -> Change runtime type -> Hardware accelerator -> GPU.

# In[12]:


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model
torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)
model_0


# Let's check out our model by doing a dummy forward pass.

# In[13]:


# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")


# ### 3.1 Making a model (TinyVGG) (script mode)
# 
# Over the past few notebooks (notebook 03 and notebook 04), we've built the TinyVGG model a few times.
# 
# So it makes sense to put the model into its file so we can reuse it again and again.
# 
# Let's put our `TinyVGG()` model class into a script called `model_builder.py` with the line `%%writefile going_modular/model_builder.py`. 

# In[14]:


get_ipython().run_cell_magic('writefile', 'going_modular/model_builder.py', '"""\nContains PyTorch model code to instantiate a TinyVGG model.\n"""\nimport torch\n\nfrom torch import nn\n\nclass TinyVGG(nn.Module):\n    """Creates the TinyVGG architecture.\n\n    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.\n    See the original architecture here: https://poloclub.github.io/cnn-explainer/\n\n    Args:\n    input_shape: An integer indicating number of input channels.\n    hidden_units: An integer indicating number of hidden units between layers.\n    output_shape: An integer indicating number of output units.\n    """\n    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:\n        super().__init__()\n        self.conv_block_1 = nn.Sequential(\n          nn.Conv2d(in_channels=input_shape, \n                    out_channels=hidden_units, \n                    kernel_size=3, \n                    stride=1, \n                    padding=0),  \n          nn.ReLU(),\n          nn.Conv2d(in_channels=hidden_units, \n                    out_channels=hidden_units,\n                    kernel_size=3,\n                    stride=1,\n                    padding=0),\n          nn.ReLU(),\n          nn.MaxPool2d(kernel_size=2,\n                        stride=2)\n        )\n        self.conv_block_2 = nn.Sequential(\n          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),\n          nn.ReLU(),\n          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),\n          nn.ReLU(),\n          nn.MaxPool2d(2)\n        )\n        self.classifier = nn.Sequential(\n          nn.Flatten(),\n          # Where did this in_features shape come from? \n          # It\'s because each layer of our network compresses and changes the shape of our inputs data.\n          nn.Linear(in_features=hidden_units*13*13,\n                    out_features=output_shape)\n        )\n    \n    def forward(self, x: torch.Tensor):\n        x = self.conv_block_1(x)\n        x = self.conv_block_2(x)\n        x = self.classifier(x)\n        return x\n        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion\n')


# Create an instance of `TinyVGG` (from the script).

# In[15]:


import torch

from going_modular import model_builder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model_1 = model_builder.TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                                hidden_units=10, 
                                output_shape=len(class_names)).to(device)
model_1


# Do a dummy forward pass on `model_1`.

# In[16]:


# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_1.eval()
with torch.inference_mode():
    pred = model_1(img_single.to(device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")


# ## 4. Creating `train_step()` and `test_step()` functions and `train()` to combine them  
# 
# Rather than writing them again, we can reuse the `train_step()` and `test_step()` functions from [notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions).
# 
# The same goes for the `train()` function we created.
# 
# The only difference here is that these functions have had docstrings added to them in [Google's Python Functions and Methods Style Guide](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).
# 
# Let's start by making `train_step()`.

# In[17]:


from typing import Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# Now we'll do `test_step()`.

# In[18]:


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# And we'll combine `train_step()` and `test_step()` into `train()`.

# In[19]:


from typing import Dict, List

from tqdm.auto import tqdm

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results


# ### 4.1 Creating `train_step()` and `test_step()` functions and `train()` to combine them (script mode)   
# 
# To create a script for `train_step()`, `test_step()` and `train()`, we'll combine their code all into a single cell.
# 
# We'll then write that cell to a file called `engine.py` because these functions will be the "engine" of our training pipeline.
# 
# We can do so with the magic line `%%writefile going_modular/engine.py`.
# 
# We'll also make sure to put all the imports we need (`torch`, `typing`, and `tqdm`) at the top of the cell.

# In[20]:


get_ipython().run_cell_magic('writefile', 'going_modular/engine.py', '"""\nContains functions for training and testing a PyTorch model.\n"""\nfrom typing import Dict, List, Tuple\n\nimport torch\n\nfrom tqdm.auto import tqdm\n\ndef train_step(model: torch.nn.Module, \n               dataloader: torch.utils.data.DataLoader, \n               loss_fn: torch.nn.Module, \n               optimizer: torch.optim.Optimizer,\n               device: torch.device) -> Tuple[float, float]:\n    """Trains a PyTorch model for a single epoch.\n\n    Turns a target PyTorch model to training mode and then\n    runs through all of the required training steps (forward\n    pass, loss calculation, optimizer step).\n\n    Args:\n    model: A PyTorch model to be trained.\n    dataloader: A DataLoader instance for the model to be trained on.\n    loss_fn: A PyTorch loss function to minimize.\n    optimizer: A PyTorch optimizer to help minimize the loss function.\n    device: A target device to compute on (e.g. "cuda" or "cpu").\n\n    Returns:\n    A tuple of training loss and training accuracy metrics.\n    In the form (train_loss, train_accuracy). For example:\n\n    (0.1112, 0.8743)\n    """\n    # Put model in train mode\n    model.train()\n\n    # Setup train loss and train accuracy values\n    train_loss, train_acc = 0, 0\n\n    # Loop through data loader data batches\n    for batch, (X, y) in enumerate(dataloader):\n        # Send data to target device\n        X, y = X.to(device), y.to(device)\n\n        # 1. Forward pass\n        y_pred = model(X)\n\n        # 2. Calculate  and accumulate loss\n        loss = loss_fn(y_pred, y)\n        train_loss += loss.item() \n\n        # 3. Optimizer zero grad\n        optimizer.zero_grad()\n\n        # 4. Loss backward\n        loss.backward()\n\n        # 5. Optimizer step\n        optimizer.step()\n\n        # Calculate and accumulate accuracy metric across all batches\n        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)\n        train_acc += (y_pred_class == y).sum().item()/len(y_pred)\n\n    # Adjust metrics to get average loss and accuracy per batch \n    train_loss = train_loss / len(dataloader)\n    train_acc = train_acc / len(dataloader)\n    return train_loss, train_acc\n\ndef test_step(model: torch.nn.Module, \n              dataloader: torch.utils.data.DataLoader, \n              loss_fn: torch.nn.Module,\n              device: torch.device) -> Tuple[float, float]:\n    """Tests a PyTorch model for a single epoch.\n\n    Turns a target PyTorch model to "eval" mode and then performs\n    a forward pass on a testing dataset.\n\n    Args:\n    model: A PyTorch model to be tested.\n    dataloader: A DataLoader instance for the model to be tested on.\n    loss_fn: A PyTorch loss function to calculate loss on the test data.\n    device: A target device to compute on (e.g. "cuda" or "cpu").\n\n    Returns:\n    A tuple of testing loss and testing accuracy metrics.\n    In the form (test_loss, test_accuracy). For example:\n\n    (0.0223, 0.8985)\n    """\n    # Put model in eval mode\n    model.eval() \n\n    # Setup test loss and test accuracy values\n    test_loss, test_acc = 0, 0\n\n    # Turn on inference context manager\n    with torch.inference_mode():\n        # Loop through DataLoader batches\n        for batch, (X, y) in enumerate(dataloader):\n            # Send data to target device\n            X, y = X.to(device), y.to(device)\n\n            # 1. Forward pass\n            test_pred_logits = model(X)\n\n            # 2. Calculate and accumulate loss\n            loss = loss_fn(test_pred_logits, y)\n            test_loss += loss.item()\n\n            # Calculate and accumulate accuracy\n            test_pred_labels = test_pred_logits.argmax(dim=1)\n            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))\n\n    # Adjust metrics to get average loss and accuracy per batch \n    test_loss = test_loss / len(dataloader)\n    test_acc = test_acc / len(dataloader)\n    return test_loss, test_acc\n\ndef train(model: torch.nn.Module, \n          train_dataloader: torch.utils.data.DataLoader, \n          test_dataloader: torch.utils.data.DataLoader, \n          optimizer: torch.optim.Optimizer,\n          loss_fn: torch.nn.Module,\n          epochs: int,\n          device: torch.device) -> Dict[str, List[float]]:\n    """Trains and tests a PyTorch model.\n\n    Passes a target PyTorch models through train_step() and test_step()\n    functions for a number of epochs, training and testing the model\n    in the same epoch loop.\n\n    Calculates, prints and stores evaluation metrics throughout.\n\n    Args:\n    model: A PyTorch model to be trained and tested.\n    train_dataloader: A DataLoader instance for the model to be trained on.\n    test_dataloader: A DataLoader instance for the model to be tested on.\n    optimizer: A PyTorch optimizer to help minimize the loss function.\n    loss_fn: A PyTorch loss function to calculate loss on both datasets.\n    epochs: An integer indicating how many epochs to train for.\n    device: A target device to compute on (e.g. "cuda" or "cpu").\n\n    Returns:\n    A dictionary of training and testing loss as well as training and\n    testing accuracy metrics. Each metric has a value in a list for \n    each epoch.\n    In the form: {train_loss: [...],\n              train_acc: [...],\n              test_loss: [...],\n              test_acc: [...]} \n    For example if training for epochs=2: \n             {train_loss: [2.0616, 1.0537],\n              train_acc: [0.3945, 0.3945],\n              test_loss: [1.2641, 1.5706],\n              test_acc: [0.3400, 0.2973]} \n    """\n    # Create empty results dictionary\n    results = {"train_loss": [],\n               "train_acc": [],\n               "test_loss": [],\n               "test_acc": []\n    }\n\n    # Loop through training and testing steps for a number of epochs\n    for epoch in tqdm(range(epochs)):\n        train_loss, train_acc = train_step(model=model,\n                                          dataloader=train_dataloader,\n                                          loss_fn=loss_fn,\n                                          optimizer=optimizer,\n                                          device=device)\n        test_loss, test_acc = test_step(model=model,\n          dataloader=test_dataloader,\n          loss_fn=loss_fn,\n          device=device)\n\n        # Print out what\'s happening\n        print(\n          f"Epoch: {epoch+1} | "\n          f"train_loss: {train_loss:.4f} | "\n          f"train_acc: {train_acc:.4f} | "\n          f"test_loss: {test_loss:.4f} | "\n          f"test_acc: {test_acc:.4f}"\n        )\n\n        # Update results dictionary\n        results["train_loss"].append(train_loss)\n        results["train_acc"].append(train_acc)\n        results["test_loss"].append(test_loss)\n        results["test_acc"].append(test_acc)\n\n    # Return the filled results at the end of the epochs\n    return results\n')


# ## 5. Creating a function to save the model
# 
# Let's setup a function to save our model to a directory.

# In[21]:


from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)


# ### 5.1 Creating a function to save the model (script mode)
# 
# How about we add our `save_model()` function to a script called `utils.py` which is short for "utilities".
# 
# We can do so with the magic line `%%writefile going_modular/utils.py`.

# In[22]:


get_ipython().run_cell_magic('writefile', 'going_modular/utils.py', '"""\nContains various utility functions for PyTorch model training and saving.\n"""\nfrom pathlib import Path\n\nimport torch\n\ndef save_model(model: torch.nn.Module,\n               target_dir: str,\n               model_name: str):\n    """Saves a PyTorch model to a target directory.\n\n    Args:\n    model: A target PyTorch model to save.\n    target_dir: A directory for saving the model to.\n    model_name: A filename for the saved model. Should include\n      either ".pth" or ".pt" as the file extension.\n\n    Example usage:\n    save_model(model=model_0,\n               target_dir="models",\n               model_name="05_going_modular_tingvgg_model.pth")\n    """\n    # Create target directory\n    target_dir_path = Path(target_dir)\n    target_dir_path.mkdir(parents=True,\n                        exist_ok=True)\n\n    # Create model save path\n    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with \'.pt\' or \'.pth\'"\n    model_save_path = target_dir_path / model_name\n\n    # Save the model state_dict()\n    print(f"[INFO] Saving model to: {model_save_path}")\n    torch.save(obj=model.state_dict(),\n             f=model_save_path)\n')


# ## 6. Train, evaluate and save the model
# 
# Let's leverage the functions we've got above to train, test and save a model to file.

# In[23]:


# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_0 
model_0_results = train(model=model_0, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS,
                        device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
save_model(model=model_0,
           target_dir="models",
           model_name="05_going_modular_cell_mode_tinyvgg_model.pth")


# ### 6.1 Train, evaluate and save the model (script mode)
# 
# Let's combine all of our modular files into a single script `train.py`.
# 
# This will allow us to run all of the functions we've written with a single line of code on the command line:
# 
# `python going_modular/train.py`
# 
# Or if we're running it in a notebook:
# 
# `!python going_modular/train.py`
# 
# We'll go through the following steps:
# 1. Import the various dependencies, namely `torch`, `os`, `torchvision.transforms` and all of the scripts from the `going_modular` directory, `data_setup`, `engine`, `model_builder`, `utils`.
#   * **Note:** Since `train.py` will be *inside* the `going_modular` directory, we can import the other modules via `import ...` rather than `from going_modular import ...`.
# 2. Setup various hyperparameters such as batch size, number of epochs, learning rate and number of hidden units (these could be set in the future via [Python's `argparse`](https://docs.python.org/3/library/argparse.html)).
# 3. Setup the training and test directories.
# 4. Setup device-agnostic code.
# 5. Create the necessary data transforms.
# 6. Create the DataLoaders using `data_setup.py`.
# 7. Create the model using `model_builder.py`.
# 8. Setup the loss function and optimizer.
# 9. Train the model using `engine.py`.
# 10. Save the model using `utils.py`. 

# In[24]:


get_ipython().run_cell_magic('writefile', 'going_modular/train.py', '"""\nTrains a PyTorch image classification model using device-agnostic code.\n"""\n\nimport os\n\nimport torch\n\nfrom torchvision import transforms\n\nimport data_setup, engine, model_builder, utils\n\n\n# Setup hyperparameters\nNUM_EPOCHS = 5\nBATCH_SIZE = 32\nHIDDEN_UNITS = 10\nLEARNING_RATE = 0.001\n\n# Setup directories\ntrain_dir = "data/pizza_steak_sushi/train"\ntest_dir = "data/pizza_steak_sushi/test"\n\n# Setup target device\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\n\n# Create transforms\ndata_transform = transforms.Compose([\n  transforms.Resize((64, 64)),\n  transforms.ToTensor()\n])\n\n# Create DataLoaders with help from data_setup.py\ntrain_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(\n    train_dir=train_dir,\n    test_dir=test_dir,\n    transform=data_transform,\n    batch_size=BATCH_SIZE\n)\n\n# Create model with help from model_builder.py\nmodel = model_builder.TinyVGG(\n    input_shape=3,\n    hidden_units=HIDDEN_UNITS,\n    output_shape=len(class_names)\n).to(device)\n\n# Set loss and optimizer\nloss_fn = torch.nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(model.parameters(),\n                             lr=LEARNING_RATE)\n\n# Start training with help from engine.py\nengine.train(model=model,\n             train_dataloader=train_dataloader,\n             test_dataloader=test_dataloader,\n             loss_fn=loss_fn,\n             optimizer=optimizer,\n             epochs=NUM_EPOCHS,\n             device=device)\n\n# Save the model with help from utils.py\nutils.save_model(model=model,\n                 target_dir="models",\n                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")\n')


# Now our final directory structure looks like:
# ```
# data/
#   pizza_steak_sushi/
#     train/
#       pizza/
#         train_image_01.jpeg
#         train_image_02.jpeg
#         ...
#       steak/
#       sushi/
#     test/
#       pizza/
#         test_image_01.jpeg
#         test_image_02.jpeg
#         ...
#       steak/
#       sushi/
# going_modular/
#   data_setup.py
#   engine.py
#   model_builder.py
#   train.py
#   utils.py
# models/
#   saved_model.pth
# ```
# 
# Now to put it all together!
# 
# Let's run our `train.py` file from the command line with:
# 
# ```
# !python going_modular/train.py
# ```
# 

# In[25]:


get_ipython().system('python going_modular/train.py')


# Woah!
# 
# Look at that!
# 
# We've just trained a model with a single line of code from the command line.
# 
# We wrote a fair of code to do so, however, now we've got our code in `.py` files we can import them and reuse them as much as we like.
# 
# For exercises and extra-curriculum for this section, refer to the [online book version of 05. PyTorch Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/#exercises).
