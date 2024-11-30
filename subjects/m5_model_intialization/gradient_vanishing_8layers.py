import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class SoftmaxRegression(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SoftmaxRegression, self).__init__(*args, **kwargs)
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.linear3 = nn.Linear(1, 1)
        self.linear4 = nn.Linear(1, 1)
        self.linear5 = nn.Linear(1, 1)
        self.linear6 = nn.Linear(1, 1)
        self.linear7 = nn.Linear(1, 1)
        self.linear8 = nn.Linear(1, 2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        init.normal_(self.linear1.weight, mean=0, std=1)
        init.normal_(self.linear2.weight, mean=0, std=1)
        init.normal_(self.linear3.weight, mean=0, std=1)
        init.normal_(self.linear4.weight, mean=0, std=1)
        init.normal_(self.linear5.weight, mean=0, std=1)
        init.normal_(self.linear6.weight, mean=0, std=1)
        init.normal_(self.linear7.weight, mean=0, std=1)
        init.normal_(self.linear8.weight, mean=0, std=1)

        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)
        init.zeros_(self.linear3.bias)
        init.zeros_(self.linear4.bias)
        init.zeros_(self.linear5.bias)
        init.zeros_(self.linear6.bias)
        init.zeros_(self.linear7.bias)
        init.zeros_(self.linear8.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        x = self.linear6(x)
        x = self.sigmoid(x)
        x = self.linear7(x)
        x = self.sigmoid(x)
        x = self.linear8(x)

        return x


def train(model, optimizer, criterion, X, y, max_epochs=1):
    for _epoch in range(max_epochs):
        xi = X[0].unsqueeze(0)
        yi = y[0].unsqueeze(0)

        optimizer.zero_grad()

        outputs = model(xi)
        print(f'outputs: {outputs.data}')

        loss = criterion(outputs, yi)
        print(f'loss: {loss}')

        loss.backward()
        optimizer.step()


# Load data
X = torch.tensor([[2.4]], dtype=torch.float32)
y = torch.tensor([0], dtype=torch.int64)

model = SoftmaxRegression()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
max_epochs = 1

train(model, optimizer, criterion, X, y, max_epochs)
