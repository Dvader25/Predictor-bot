import random
import json
import torch
from weather import construct
from stock import get_predict
from train import neural_network
from chatbot import bagOfWords, tokenize

with open('intent.json', 'r') as f:
    intents = json.load(f)
print("PredictorBot: Let's chat! You can end the conversation by saying 'quit'")

data = torch.load('data.pth')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = neural_network(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

while True:
    s = input("You: ")
    if s == 'quit':
        break

    s = tokenize(s)
    x = bagOfWords(s, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print("PredictorBot:", random.choice(intent['responses']))

        if tag == "weather":
            city = input("You: ")
            print("PredictorBot: Weather code?")
            days = int(input())
            print(construct(city, days))

        if tag == "stock":
            s = input("You: ")
            get_predict(s)


    else:
        print("PredictorBot: I do not understand...")
