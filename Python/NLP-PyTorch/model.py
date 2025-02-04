import torch
from torch.nn import *


class NeuralNet(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.linear3 = Linear(hidden_size, num_classes)
        self.activation = ReLU()

    def forward(self, X):
        preds = self.activation(self.linear1(X))
        preds = self.activation(self.linear2(preds))
        preds = self.linear3(preds)
        return preds
