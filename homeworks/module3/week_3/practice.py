import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset_man import dataset_manager

def main():
    housing_data = dataset_manager.load_dataset('m3.ex.ensemble.learning.20240903.Housing')
    df = pd.DataFrame(housing_data)
    print(df.head())
    print(df.info())

    categorical_cols = df.select_dtypes(include=['object']).columns.to_list()
    print(categorical_cols)

    # encode categorical data
    ordinal_encoder = OrdinalEncoder()
    encoded_categorical_cols = ordinal_encoder.fit_transform(df[categorical_cols])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_cols, columns=categorical_cols)
    numerical_df = df.drop(categorical_cols, axis=1)
    encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)
    print(encoded_df.head())
    print(encoded_df.shape)

    # standardization
    normalizer = StandardScaler()
    dataset_arr = normalizer.fit_transform(encoded_df)
    print(dataset_arr[0])

    # split train, test data
    X, y = dataset_arr[:, 1:], dataset_arr[:, 0]
    print('XXXXX:\n', X)
    print('yyyyy:\n', y)

    test_size = 0.3
    random_state = 1
    is_shuffle = True
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffle)

    # training model
    # random forest
    regressor = RandomForestRegressor(
        random_state=random_state
    )

    regressor.fit(X_train, y_train)

    # adaboost regressor
    ada_regressor = AdaBoostClassifier(
        random_state=random_state
    )

    # ada_regressor.fit(X_train, y_train)

    # gradient boost
    grad_regressor = GradientBoostingRegressor(
        random_state=random_state
    )

    # grad_regressor.fit(X_train, y_train)

    # validation
    y_pred = regressor.predict(X_val)
    # ada_y_pred = ada_regressor.predict(X_val)
    # grad_y_pred = grad_regressor.predict(X_val)

    print('---validation on Random Forest')
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    print('Evaluation results on validation set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    # print('---validation on Adaboost')
    # ada_mae = mean_absolute_error(y_val, ada_y_pred)
    # ada_mse = mean_squared_error(y_val, ada_y_pred)

    # print('Evaluation results on validation set:')
    # print(f'Mean Absolute Error: {ada_mae}')
    # print(f'Mean Squared Error: {ada_mse}')

    # print('---validation on Gradient Boost')
    # grad_mae = mean_absolute_error(y_val, grad_y_pred)
    # grad_mse = mean_squared_error(y_val, grad_y_pred)

    # print('Evaluation results on validation set:')
    # print(f'Mean Absolute Error: {grad_mae}')
    # print(f'Mean Squared Error: {grad_mse}')

if __name__ == "__main__":
    main()
