import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate(model_classifier, X_val, y_val):
    with torch.no_grad():
        y_pred = model_classifier(X_val)

    y_pred = torch.argmax(y_pred, dim=1)

    return sum(y_pred == y_val) / len(y_val)


data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.4)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test)

model_classifier = nn.Sequential(
    nn.Linear(4, 8),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

criterion = nn.CrossEntropyLoss()
lr = 1e-2
momentum = 0.8
optimizer = optim.SGD(model_classifier.parameters(),
                      lr=lr, momentum=momentum, nesterov=True)

epochs = 50
losses = []

for epoch in range(epochs):
    epoch_loss = []

    for xi_train, yi_train in zip(X_train, y_train):
        y_pred = model_classifier(xi_train)
        loss = criterion(y_pred, yi_train)
        epoch_loss.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    losses.append(avg_loss)
    acc = evaluate(model_classifier, X_val, y_val)
    print(f'{avg_loss} -- {acc}')

with torch.no_grad():
    y_pred = model_classifier(X_test)

y_pred = torch.argmax(y_pred, dim=1)
print(sum(y_pred == y_test)/len(y_test))
