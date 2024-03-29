import random
import json
import torch
from modelo import Red
from nltk_utils import bolsa_de_palabras, tokenizacion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open ('intenciones.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = Red(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "JefeVirtual"
print("Hola, resolvere tu duda! escribe 'salir' para salir")
while True:
    sentence = input ('You: ')
    if sentence == 'salir':
        break

    sentence = tokenizacion(sentence)
    x = bolsa_de_palabras(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents ["intenciones"]:
            if tag == intent["etiqueta"]:
                print(f"{bot_name}: {random.choice(intent['respuestas'])}")
    else:
        print(f"{bot_name}:Me esta costando entender tu pregunta...")
