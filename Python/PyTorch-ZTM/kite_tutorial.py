#!/usr/bin/env python
# coding: utf-8

# ![Logo](https://kite.com/kite-public/kite-plus-jlab-scaled.png)
# 
# ### Welcome to Kite's JupyterLab extension tutorial
# 
# Kite gives you **ML-powered autocompletions** and **rich documentation** inside JupyterLab. This guide will teach you everything you need to know about Kite in 5 minutes or less.
# 
# > 💡 _**Tip:** You can open this file at any time with the command `Kite: Open Tutorial` in JupyterLab's command palette._
# 
# #### Before we start...
# 
# Make sure that the Kite icon at the bottom of the window reads `Kite: ready`.
# 
# ![Kite icon](https://kite.com/kite-public/kite-status.png)
# 
# * If it says `Kite: not running`, please start the Kite Engine first.
# * If it says `Kite: not installed`, please [download and install Kite](https://kite.com/download) first.

# #### Part 1: Autocompletions
# 
# **Step 1a**<br/>
# Run the code cell below with all the necessary imports 👇

# In[ ]:


# Run me!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# **Step 1b**<br/>
# Let's try typing out some code to plot a sine graph. As you type, Kite will automatically show you completions for what you're going to type next.
# 
# ![Autocompletions](https://www.kite.com/kite-public/kite-jlab-autocompletions.gif)
# 
# > 💡 _**Tip:** You can turn completions docs on and off in JupyterLab's command palette with the command `Kite: Toggle Docs Panel`._
# 
# > 💡 _**Tip:** The starred completions ★ are from Kite Pro. You can [start your free Kite Pro trial](https://www.kite.com/pro/trial/) anytime. Afterwards, if you choose not to upgrade, you can still use Kite 100% for free._
# 
# Try typing out the code yourself to see Kite's autocompletions in action.<br/>
# 
# ```python
# x = np.linspace(-np.pi, np.pi, 50)
# y = np.sin(x)
# plt.plot(x, y)
# ```
# 
# Type this code in the cell below 👇

# In[ ]:


# Put code in me



# #### Part 2: Manual completions
# 
# You can still use JupyterLab's builtin kernel completions. These are particularly useful when you need to access a `DataFrame`'s column names.
# 
# **Step 2a**<br/>
# First, run the code cell below to get some sample data to store in a `DataFrame` 👇

# In[ ]:


# Run me!
url = 'https://kite.com/kite-public/iris.csv'
df = pd.read_csv(url)
df.head()


# **Step 2b**<br/>
# Let's plot a scatter graph of sepal length vs. sepal width. When you are accessing a `DataFrame`'s columns, you'll still need to hit `tab` to request completions from the kernel.
# 
# ![Manual completions](https://www.kite.com/kite-public/kite-jlab-manual-completions.gif)
# 
# Try requesting kernel completions yourself.
# 
# ```python
# plt.scatter(df['sepal_length'], df['sepal_width'])
# ```
# 
# Type this code in the cell below, making sure to hit `tab` when you are filling in the column names 👇

# In[ ]:


# Put code in me



# #### Part 3: Copilot Documentation
# 
# If you've enabled "docs following cursor" in the Copilot, the Copilot will automatically update with the documentation of the identifier underneath your cursor.
# 
# ![Autosearch](https://www.kite.com/kite-public/kite-jlab-autosearch.gif)
# 
# **Step 3a**<br/>
# Try it yourself! Just click around in the code cells of this notebook and see the Copilot update automatically.

# #### The End
# 
# Now you know everything you need to know about Kite's JupyterLab plugin. Kite is under active development and we expect to ship improvements and more features in the near future.
# 
# In the meantime, if you experience bugs or have feature requests, feel free to open an issue in our [public GitHub repo](https://github.com/kiteco/issue-tracker).
# 
# Happy coding!
