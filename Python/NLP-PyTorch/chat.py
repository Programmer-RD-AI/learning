import random
import json
import torch
from model import NeuralNet
from nltk_utils import *
device = "cuda"
with open('intents.json','r') as f:
    intents = json.load(f)
FILE = 'data.pth'
data = torch.load(FILE)
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name = 'Programmer-RD-AI'
print('Lets chat ! type "quit" to exit')
while True:
    sentence = input('You : ')
    if sentence == 'quit':
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    pred = model(X)
    pred_ = pred.clone()
    _,pred = torch.max(pred,dim=1)
    tag = tags[pred.item()]
    probs = torch.softmax(pred_,dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: IDK..')
