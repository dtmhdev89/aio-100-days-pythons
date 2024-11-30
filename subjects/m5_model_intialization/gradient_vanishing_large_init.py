import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class SoftmaxRegression(nn.Module):
    def __init__(self) -> None:
        super(SoftmaxRegression, self).__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights_and_bias()

    def _init_weights_and_bias(self):
        init.normal_(self.linear1.weight, mean=0, std=10)
        init.normal_(self.linear2.weight, mean=0, std=10)
        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)

        return x


def train(model, optimizer, criterion, X, y, max_epochs=1):
    for _epoch in range(max_epochs):
        xi = X[0].unsqueeze(0)
        yi = y[0].unsqueeze(0)

        print(f'xi: {xi}')
        print(f'yi: {yi}')
        optimizer.zero_grad()

        outputs = model(xi)
        loss = criterion(outputs, yi)
        print(f'outputs: {outputs.data}')
        print(f'loss: {loss}')

        loss.backward()
        optimizer.step()


# Load data
X = torch.tensor([[2.4]], dtype=torch.float32)
y = torch.tensor([0], dtype=torch.int64)

model = SoftmaxRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
max_epoch = 1

print(model.linear1.weight)
train(model, optimizer, criterion, X, y, max_epoch)
