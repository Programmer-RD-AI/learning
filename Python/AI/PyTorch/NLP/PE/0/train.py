from tqdm import tqdm
from torch.optim import *
from model import *
import torch, torchvision
import numpy as np
import json
from nltk_utils import *
from torch.nn import *
from torch.utils.data import Dataset, DataLoader

with open("./intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", ".", "@", "#", "$", "%", "^", "&", "*"]
new_all_words = []
for w in all_words:
    if w not in ignore_words:
        new_all_words.append(stem(w))
all_words = new_all_words
all_words = sorted(set(all_words))
tags = sorted(set(tags))
X_train = []
y_train = []
for patter_sentence, tag in xy:
    bag = bag_of_words(patter_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
print(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
device = "cuda"
epochs = 1000
epochs_iter = tqdm(range(epochs))
dataset = ChatDataSet()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
for epoch in epochs_iter:
    for words, labels in train_loader:
        print(labels)
        words = words.to(device)
        labels = labels.to(device)
        preds = model(words)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epochs_iter.set_description(f'Loss - {loss.item()}')
data = {
    'model_state':model.state_dict(),
    'input_size':input_size,
    'output_size':output_size,
    'hidden_size':hidden_size,
    'all_words':all_words,
    'tags':tags
}

FILE = 'data.pth'
torch.save(data,FILE)
