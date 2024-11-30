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
        self.prelu = nn.PReLU()
        self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        # Init weights with normal distribution
        init.normal_(self.linear1.weight, mean=0, std=3)
        init.normal_(self.linear2.weight, mean=0, std=3)

        # Optional init bias with zero
        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.prelu(x)
        x = self.linear2(x)

        return x
    

def train(model, criterion, optimizer, X, y, max_epochs=1):
    for _epoch in range(max_epoch):
        xi = X[0].unsqueeze(0)
        yi = y[0].unsqueeze(0)

        print(f'xi: {xi}')
        print(f'yi: {yi}')

        optimizer.zero_grad()

        outputs = model(xi)
        print(f'outputs: {outputs}')
        print(f'outputs: {outputs.data}')

        loss = criterion(outputs, yi)
        print(f'loss: {loss}')

        # Backward pass and optimizer
        loss.backward()
        optimizer.step()


# Load data
X = torch.tensor([[20.4]], dtype=torch.float32)
y = torch.tensor([0], dtype=torch.int64)

model = SoftmaxRegression()

print(model.linear1.weight, model.linear1.bias)
print(model.linear2.weight, model.linear2.bias)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=10)

# Training loop
max_epoch = 1

train(model, criterion, optimizer, X, y, max_epoch)
