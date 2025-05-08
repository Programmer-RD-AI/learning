#!/usr/bin/env python
# coding: utf-8

# # But what is a neural network? | Chapter 1, Deep learning
# 
# - Neuron -> a thing that hold a number, the image makes up the first layer
# - Output layer -> Contains the no. of classes and it outputs the probability that things are which
# - Hidden layers breaks down the detection of different images to its characteristics
# - weights are the connections between neurons
# - bias is another value that can be changed as a parameter and it lowers or highers the threshold at which point an neutron is fired
# - a(1)(n) = sigmoid(a(0)(n)*w(0)...+bias)

# # Gradient descent, how neural networks learn | Chapter 2, Deep learning
# 
# - cost function (loss function) see the difference between the expected answer and the give answer and calculates a value that shows how bad the predictions are
# ![Gradient descent, how neural networks learn _ Chapter 2, Deep learning - YouTube.png](attachment:a8f8fbcd-f299-42e0-a250-3df657d051db.png)
# - different weights has different power or use when changed

# # What is backpropagation really doing? | Chapter 3, Deep learning
# 
# An algorithm the calculate the gradient
# 
# when a result is given out the data is checked
# 
# so for example is we are predicting a 2 and the model gives out probability for other possibilities then we lower the weights or change the activation or change the base so as that the neurons we want to give 2 fire more and the others fire less frequently 
# 
# ![(1) What is backpropagation really doing_ _ Chapter 3, Deep learning - YouTube.png](attachment:e0969d59-2bf0-431b-809f-30c51d2edffc.png)
# 
# we go through the above process for many images an average out the change

# In[ ]:




