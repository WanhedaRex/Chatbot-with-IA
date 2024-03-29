import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bolsa_de_palabras, tokenizacion, stem
from modelo import Red

with open ('intenciones.json','r') as f:
    intents = json.load(f)

todas_las_palabras = []
etiquetas = []
xy = []

for intent  in intents['intenciones']:
    etiqueta = intent["etiqueta"]
    etiquetas.append(etiqueta)
    for patron in intent['patrones']:
        w = tokenizacion(patron)
        todas_las_palabras.extend(w)
        xy.append((w,etiqueta))

ignorar_palabras = ['?',',','.','¡','¿','!','-','_']
todas_las_palabras = [stem(w) for w in todas_las_palabras if w not in ignorar_palabras]


todas_las_palabras = sorted(set(todas_las_palabras))
etiquetas = sorted(set(etiquetas))

X_train = []
Y_train = []
for (oracion_patron,etiqueta) in xy:
    bolsa = bolsa_de_palabras(oracion_patron, todas_las_palabras)
    X_train.append(bolsa)
    label = etiquetas.index(etiqueta)
    Y_train.append(label)

X_train_train = np.array(X_train)
Y_train_train = np.array(Y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(etiquetas)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train_train


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelo = Red(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for (palabras, labels) in train_loader:
        palabras = palabras.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = modelo(palabras)

        loss = criterion(outputs, labels)
        
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')
data = {
    "model_state": modelo.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": todas_las_palabras,
    "tags": etiquetas
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')