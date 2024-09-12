import json
import random
import torch
from model import NeuralN
from word_bag import words_of_bag,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Smith"
print("Hey..!, I'm Smith")

while True:
    sentence = input('You: ')
    if sentence == 'bye smith':
        break

    sentence = tokenize(sentence)
    X = words_of_bag(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    prob = torch.softmax(output, dim=1)
    prob = prob[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    
    else:
        print(f"{bot_name}: I don't understand. . . .")