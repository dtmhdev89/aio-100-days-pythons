import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from dataset_man import dataset_manager

def main():
    sns.set()
    plt.style.use('fivethirtyeight')

    dm = dataset_manager
    data = pd.DataFrame(dm.load_dataset('m3.decision.tree.regression.20240906.MSFT'))
    print(data.head())

    plt.figure(figsize=(10, 4))
    plt.title("Microsoft Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.plot(data["Close"])
    plt.show()

    x = data[["Open", "High", "Low"]]
    y = data["Close"]
    x = x.to_numpy()
    y = y.to_numpy()
    y = y.reshape(-1, 1)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    pred_data = pd.DataFrame(data={"Predicted Rate": ypred})
    print(pred_data.head())
    print(ytest[:5])
    print(mean_squared_error(ytest, ypred))
    print(r2_score(ytest, ypred))

if __name__ == "__main__":
    main()
