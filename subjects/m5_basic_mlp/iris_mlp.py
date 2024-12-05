from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt


def convert_to_tensor(x, train=True):
    dtype = torch.float32 if train else torch.long

    return torch.tensor(x, dtype=dtype)


def evaluate(model_classifier, X_valid, y_valid):
    with torch.no_grad():
        y_pred = model_classifier(X_valid)

    y_pred = torch.argmax(y_pred, dim=1)

    return sum(y_pred == y_valid) / len(y_valid)


def train(model_classifier, optimizer,
          criterion, X_train, X_val, y_train, y_val):
    num_epochs = 20
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for xi_train, yi_train in zip(X_train, y_train):
            optimizer.zero_grad()
            yi_pred = model_classifier(xi_train)
            loss = criterion(yi_pred, yi_train)
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        losses.append(avg_loss)
        acc = evaluate(model_classifier=model_classifier, X_valid=X_val, y_valid=y_val)
        print(f'{avg_loss} -- {acc}')

    return losses

if __name__ == "__main__":
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.4,
        random_state=7
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=0.5,
        random_state=7
    )

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = convert_to_tensor(X_train)
    X_val = convert_to_tensor(X_val)
    X_test = convert_to_tensor(X_test)
    y_train = convert_to_tensor(y_train, train=False)
    y_val = convert_to_tensor(y_val, train=False)
    y_test = convert_to_tensor(y_test, train=False)

    print(f'Xtrain shape: {X_train.shape}')
    print(f'y labels: {y_train.unique()}')

    model_classifier = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )

    # summary(model_classifier, (1, 4))

    criterion = nn.CrossEntropyLoss()

    input = X_train[0]
    output = model_classifier(input)
    loss = criterion(output, y_train[0])
    print(loss.item())

    lr = 1e-2
    optimizer = optim.SGD(
        model_classifier.parameters(),
        lr=lr
    )

    print(evaluate(model_classifier=model_classifier, X_valid=X_val, y_valid=y_val))

    losses = train(model_classifier=model_classifier,
                   optimizer=optimizer,
                   criterion=criterion,
                   X_train=X_train,
                   y_train=y_train,
                   X_val=X_val,
                   y_val=y_val)

    test_accuracy = evaluate(model_classifier=model_classifier,
                        X_valid=X_test, y_valid=y_test)

    _fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, 21), losses)
    plt.xticks(range(1, 21))
    plt.show()
