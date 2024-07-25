import torch
import torch.nn as nn

class ReLUActivateFunction(nn.Module):
    def __init__(self):
        super(ReLUActivateFunction, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

class SigmoidActivateFunction(nn.Module):
    def __init__(self):
        super(SigmoidActivateFunction, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)
    
class ReLUActivateFunctionV2():
    def __init__(self, vector) -> None:
        self.vector = vector

    def calculate(self):
        return [round(max(0, x), 4) for x in self.vector]

class SigmoidActivateFunctionV2():
    def __init__(self, vector) -> None:
        self.vector = vector

    def calculate(self):
        epsilon = 2.71828
        formula = lambda x: round(1 / ( 1 + epsilon ** (-x)), 4)
        return [ formula(x) for x in self.vector]

def main():
    tensor = torch.tensor([1, -5, 1.5, 2.7, -5])

    print(ReLUActivateFunction()(tensor))
    print(SigmoidActivateFunction()(tensor))

    print(ReLUActivateFunctionV2(tensor.tolist()).calculate())
    print(SigmoidActivateFunctionV2(tensor.tolist()).calculate())

if __name__ == "__main__":
    main()
