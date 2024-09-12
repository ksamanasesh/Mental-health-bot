import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from word_bag import tokenize, stem, words_of_bag
from model import NeuralN

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore = ['?','.',',','!']
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_Train = []
y_Train = []

for (pattern_senctence, tag) in xy:
    bag = words_of_bag(pattern_senctence,all_words)
    X_Train.append(bag)

    label = tags.index(tag)
    y_Train.append(label)

X_Train = np.array(X_Train)
y_Train = np.array(y_Train)

class ChatDataSet(Dataset):
    
    def __init__(self):
        self.n_samples = len(X_Train)
        self.x_train = X_Train
        self.y_train = y_Train

    def __getitem__(self, index):
        return self.x_train[index] , self.y_train[index]

    def __len__(self):
        return self.n_samples

#hyperparamerters    
hidden_size = 8
output_size = len(tags)
input_size = len(X_Train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralN(input_size, hidden_size, output_size).to(device)

#loss and optimzer 
criterion = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimzer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data,FILE)
print(f'Training Completed. File saved to {FILE}')