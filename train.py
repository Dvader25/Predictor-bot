#training the neural network for natural language processing 

import json
import numpy as np
from chatbot import tokenize, stem, bagOfWords
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intent.json', 'r') as f:
    intents = json.load(f)

# print(intents)
all_words = []
tags = []
xy = []

for i in intents['intents']:
    tag = i['tag']
    tags.append(tag)
    for p in i['patterns']:
        p = tokenize(p)
        all_words.extend(p)
        xy.append((p,tag))

ignore = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))

x_train = []
y_train = []

for (p, t) in xy:
    x_train.append(bagOfWords(p, all_words))
    y_train.append(tags.index(t))

x_train = np.array(x_train)
y_train = np.array(y_train)

class chat_data(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = chat_data()
train_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

class neural_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(neural_network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = neural_network(len(x_train[0]), 8, len(tags))

cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

print("Training the model ...")
for i in range(1000): #epochs
    for (w, l) in train_loader:
        # w = w.to(device)
        # l = l.to(device)
        l = l.to(dtype=torch.long)

        output = model(w)
        loss = cost(output, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if (i+1) % 100 == 0:
    #     print('epoch ', i+1, 'loss = ', loss)

data = {
    "model_state": model.state_dict(),
    "input_size": len(all_words),
    "output_size": len(tags),
    "hidden_size": 8,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, 'data.pth')










