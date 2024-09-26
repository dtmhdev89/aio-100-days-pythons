import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
    mean_squared_error

from dataset_man import dataset_manager

def evaluate_model(y_test, prediction):
  print(f"MAE: {mean_absolute_error(y_test, prediction)}")
  print(f"MSE: {mean_squared_error(y_test, prediction)}")
  print(f"MAPE: {mean_absolute_percentage_error(y_test, prediction)}")

def plot_predictions(testing_dates, y_test, prediction):
  df_test = pd.DataFrame({"date": testing_dates, "actual": y_test, "prediction": prediction })
  _figure, ax = plt.subplots(figsize=(10, 5))
  df_test.plot(ax=ax, label="Actual", x="date", y="actual")
  df_test.plot(ax=ax, label="Prediction", x="date", y="prediction")
  plt.legend(["Actual", "Prediction"])
  plt.show()

def main():
    lodon_energy_data = dataset_manager.load_gdrive_dataset('1gAgOxOerjJdtuGAA6yInw1auPiqXn5u0')

    df = pd.DataFrame(lodon_energy_data)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isna().sum())

    df_avg_consumption = df.groupby("Date")["KWH"].mean()
    df_avg_consumption = pd.DataFrame({"date": df_avg_consumption.index.tolist(), "consumption": df_avg_consumption.values.tolist()})
    df_avg_consumption["date"] = pd.to_datetime(df_avg_consumption["date"])
    print(f"From: {df_avg_consumption['date'].min()}")
    print(f"To: {df_avg_consumption['date'].max()}")

    df_avg_consumption.plot(x="date", y="consumption")
    plt.show()

    df_avg_consumption.query("date > '2012-01-01' & date < '2013-01-01'").plot(x="date", y="consumption")
    plt.show()

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

    _figure, ax = plt.subplots(figsize=(20, 5))
    training_data.plot(ax=ax, label="Training", x="date", y="consumption")
    testing_data.plot(ax=ax, label="Testing", x="date", y="consumption")
    plt.show()

    # Dropping unnecessary `date` column
    training_data = training_data.drop(columns=["date"])
    testing_dates = testing_data["date"]
    testing_data = testing_data.drop(columns=["date"])

    columns_selection = ["day_of_week", "day_of_year", "month", "quarter", "year"]
    label_column = "consumption"
    X_train = training_data[columns_selection]
    y_train = training_data[label_column]

    X_test = testing_data[columns_selection]
    y_test = testing_data[label_column]

    # XGBoost
    cv_split = TimeSeriesSplit(n_splits=4, test_size=100)
    model = XGBRegressor()
    parameters = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [100, 300],
        "colsample_bytree": [0.3]
    }


    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters, verbose=2, n_jobs=2)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)

    # Evaluating GridSearch results
    prediction = grid_search.predict(X_test)
    plot_predictions(testing_dates, y_test, prediction)
    evaluate_model(y_test, prediction)

    # Add more data (more features in this case) to improve performance
    london_weather = dataset_manager.load_dataset('m3.xgboost.20241809.london_weather')
    df_weather = pd.DataFrame(london_weather)
    print(df_weather.isna().sum())
    print(df_weather.head())

    # Parsing dates
    df_weather["date"] = pd.to_datetime(df_weather["date"], format="%Y%m%d")

    # Filling missing values through interpolation
    df_weather = df_weather.ffill()

    # Enhancing consumption dataset with weather information
    df_avg_consumption = df_avg_consumption.merge(df_weather, how="inner", on="date")
    print(df_avg_consumption.head())

    training_mask = df_avg_consumption["date"] < "2013-07-28"
    training_data = df_avg_consumption.loc[training_mask]
    print(training_data.shape)

    testing_mask = df_avg_consumption["date"] >= "2013-07-28"
    testing_data = df_avg_consumption.loc[testing_mask]
    print(testing_data.shape)

    # Dropping unnecessary `date` column
    training_data = training_data.drop(columns=["date"])
    testing_dates = testing_data["date"]
    testing_data = testing_data.drop(columns=["date"])

    columns_selection = ["day_of_week", "day_of_year", "month", "quarter", "year",\
                            "cloud_cover", "sunshine", "global_radiation", "max_temp",\
                            "mean_temp", "min_temp", "precipitation", "pressure",\
                            "snow_depth"]

    X_train = training_data[columns_selection]
    y_train = training_data[label_column]

    X_test = testing_data[columns_selection]
    y_test = testing_data[label_column]

    # XGBoost
    cv_split = TimeSeriesSplit(n_splits=4, test_size=100)
    model = XGBRegressor()
    parameters = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [100, 300],
        "colsample_bytree": [0.3]
    }


    grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters, n_jobs=2)
    grid_search.fit(X_train, y_train)

    print('Best Parameter: ', grid_search.best_params_)

    # Evaluating GridSearch results
    prediction = grid_search.predict(X_test)
    plot_predictions(testing_dates, y_test, prediction)
    evaluate_model(y_test, prediction)

if __name__ == "__main__":
    main()
