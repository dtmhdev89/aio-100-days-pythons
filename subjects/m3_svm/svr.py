import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset_man import dataset_manager

def main():
    auto_insurance_data = dataset_manager.load_dataset('m3.svm.20240914.auto-insurance', read_file_options={'names': ['n_claims', 'total_payment']})

    df = pd.DataFrame(auto_insurance_data)

    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.shape)

    df.boxplot(column='n_claims')
    df.plot(kind='scatter', x='n_claims', y='total_payment')

    plt.show()

    normalizer = StandardScaler()
    df_normalized = normalizer.fit_transform(df)

    X, y = df_normalized[:, 0], df_normalized[:, 1]
    X = X.reshape(-1, 1)

    test_size = 0.3
    random_state = 1
    is_shuffle = True
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    print(f'Number of training samples: {X_train.shape[0]}')
    print(f'Number of val samples: {X_val.shape[0]}')

    regressor = SVR()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_val)
    mae = mean_absolute_error(y_pred, y_val)
    mse = mean_squared_error(y_pred, y_val)

    print('Evaluation results on validation set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')


if __name__ == "__main__":
    main()
