import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from dataset_man import dataset_manager

def evaluate_model(y_test, prediction):
    print(f"MAE: {mean_absolute_error(y_test, prediction)}")
    print(f"MSE: {mean_squared_error(y_test, prediction)}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, prediction)}")

def plot_predictions(testing_dates, y_test, prediction):
    df_test = pd.DataFrame({"date": testing_dates, "actual": y_test, "prediction": prediction })
    figure, ax = plt.subplots(figsize=(10, 5))
    df_test.plot(ax=ax, label="Actual", x="date", y="actual")
    df_test.plot(ax=ax, label="Prediction", x="date", y="prediction")
    plt.legend(["Actual", "Prediction"])
    plt.show()

def main():
    london_energy_data = dataset_manager.load_gdrive_dataset('1gAgOxOerjJdtuGAA6yInw1auPiqXn5u0')
    df = pd.DataFrame(london_energy_data)
    # df = pd.read_csv("/content/drive/MyDrive/AIO2024/london_energy.csv")
    print(df.isna().sum())
    print(df.head())

    df_avg_consumption = df.groupby("Date")["KWH"].mean()
    df_avg_consumption = pd.DataFrame({"date": df_avg_consumption.index.tolist(), "consumption": df_avg_consumption.values.tolist()})
    df_avg_consumption["date"] = pd.to_datetime(df_avg_consumption["date"])
    print(f"From: {df_avg_consumption['date'].min()}")
    print(f"To: {df_avg_consumption['date'].max()}")

    df_avg_consumption.plot(x="date", y="consumption")

    df_avg_consumption.query("date > '2012-01-01' & date < '2013-01-01'").plot(x="date", y="consumption")

    df_avg_consumption["day_of_week"] = df_avg_consumption["date"].dt.dayofweek
    df_avg_consumption["day_of_year"] = df_avg_consumption["date"].dt.dayofyear
    df_avg_consumption["month"] = df_avg_consumption["date"].dt.month
    df_avg_consumption["quarter"] = df_avg_consumption["date"].dt.quarter
    df_avg_consumption["year"] = df_avg_consumption["date"].dt.year

    print(df_avg_consumption.head())

    training_mask = df_avg_consumption["date"] < "2013-07-28"
    training_data = df_avg_consumption.loc[training_mask]
    print(training_data.shape)

    testing_mask = df_avg_consumption["date"] >= "2013-07-28"
    testing_data = df_avg_consumption.loc[testing_mask]
    print(testing_data.shape)

    figure, ax = plt.subplots(figsize=(20, 5))
    training_data.plot(ax=ax, label="Training", x="date", y="consumption")
    testing_data.plot(ax=ax, label="Testing", x="date", y="consumption")

    # Dropping unnecessary `date` column
    training_data = training_data.drop(columns=["date"])
    testing_dates = testing_data["date"]
    testing_data = testing_data.drop(columns=["date"])

    X_train = training_data[["day_of_week", "day_of_year", "month", "quarter", "year"]]
    y_train = training_data["consumption"]

    X_test = testing_data[["day_of_week", "day_of_year", "month", "quarter", "year"]]
    y_test = testing_data["consumption"]

    # XGBoost
    cv_split = TimeSeriesSplit(n_splits=4, test_size=100)
    model = XGBRegressor()
    parameters = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [100, 300],
        "colsample_bytree": [0.3]
    }


    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
    grid_search.fit(X_train, y_train)

    # Evaluating GridSearch results
    prediction = grid_search.predict(X_test)
    plot_predictions(testing_dates, y_test, prediction)
    evaluate_model(y_test, prediction)

    print(prediction)

    # Initialize data to lists.
    data = [{'day_of_week': 6, 'day_of_year': 209, 'month': 7, 'quarter': 3, 'year':2013}]

    # Creates DataFrame.
    df = pd.DataFrame(data)
    prediction = grid_search.predict(df)
    print(prediction)

    plt.show()


if __name__ == "__main__":
    main()
