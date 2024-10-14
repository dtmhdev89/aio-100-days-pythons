import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from prepare_data import BTCDataset

def predict(X, w, b):
    return X.dot(w) + b

def gradient(y_hat, y, x):
    loss = y_hat - y
    dw = x.T.dot(loss) / len(y)
    db = np.sum(loss) / len(y)
    cost = np.sum(loss**2) / (2*len(y))

    return (dw, db, cost)

def compute_loss(y_hat, y):
    return ((y_hat - y)**2) / 2

def update_weight(w, b, lr, dw, db):
    w_new = w - dw*lr
    b_new = b - db*lr

    return (w_new, b_new)

def linear_regression_vectorized(X, y, lr=1e-2, num_iterations=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features).reshape(-1, 1)
    b = 0
    losses = []

    for _epoch in range(num_iterations):
        y_hat = predict(X, w, b)
        
        # loss = compute_loss(y_hat, y)

        dw, db, cost = gradient(y_hat, y, X)

        w, b = update_weight(w, b, lr, dw, db)

        losses.append(cost)

    return (w, b, losses)


def main():
    df = pd.DataFrame(BTCDataset().df)
    print(df.info())
    print(df.columns)
    print(df.head())
    df['date'] = pd.to_datetime(df['date'])
    date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())
    print(date_range)

    df.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
    df.index = df['date']
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    print(df.head())

    # grouped_close_price_df = df[['year', 'close']].groupby('year').mean()
    # print(grouped_close_price_df)

    unique_years = sorted(df['year'].unique())
    print(unique_years)

    # for year in unique_years:
    #     date_range_in_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31',freq='D')
    #     year_month_day = pd.DataFrame({
    #         'year': date_range_in_year.year,
    #         'month': date_range_in_year.month,
    #         'day': date_range_in_year.day,
    #         'date_x': date_range_in_year.date})
    #     merged_data = pd.merge(year_month_day, df, on=['year', 'month', 'day'], how='left')
    #     print(merged_data.head(100))
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(merged_data['date_x'], merged_data['close'])
    #     plt.title(f'Bitcoin Closing Prices - {year}')
    #     plt.xlabel('Date')
    #     plt.ylabel('Closing Price (USD)')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()

    # candle stick plot
    df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31')]
    df_filtered['date'] = df_filtered['date'].map(mdates.date2num)

    fig, ax = plt.subplots(figsize=(20, 6))
    candlestick_ohlc(ax, df_filtered[['date', 'open', 'high', 'low', 'close']].values, width=0.6, colorup='g', colordown='r')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.title('Bitcoin Candle Stick Chart (2019-2022)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.savefig('bitcoin_candle_stick_2019_2022.pdf')
    # plt.show()

    # x_columns = ['open', 'high', 'low', 'Volume BTC', 'Volume USD', 'year', 'month', 'day']
    x_columns = ['open', 'high', 'low']
    y_column = ['close']

    X = df[x_columns].values
    y = df[y_column].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    print(X_train.shape, y_train.shape)

    w, b, losses = linear_regression_vectorized(X_train, y_train, lr=1e-2, num_iterations=200)

    y_pred = predict(X_test, w, b)
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    mae = np.mean(np.abs(y_pred - y_test))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    y_train_pred = predict(X_train, w, b)
    train_accuracy = r2_score(y_train, y_train_pred)

    test_accuracy = r2_score(y_test, y_pred)

    print('Root Mean Squared Error (RMSE): ', round(rmse, 4))
    print('Mean Absolute Error (MAE): ', round(mae, 4))
    print('Training Accuracy (R-squared): ', round(train_accuracy, 4))
    print('Testing Accuracy (R-squared): ', round(test_accuracy, 4))

    # inference
    df_2019_q1 = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2019-04-01')]

    X_2019_q1 = df_2019_q1.loc[:, ['open', 'high', 'low']]
    y_2019_q1_actual = df_2019_q1['close']

    y_2019_q1_pred = predict(X_2019_q1, w, b)

    plt.figure(figsize=(12, 6))
    plt.plot(df_2019_q1['date'], y_2019_q1_actual, label='Actual Close Price', marker='o')
    plt.plot(df_2019_q1['date'], y_2019_q1_pred, label='Predicted Close Price', marker='x')
    plt.title('Actual Vs Predicted Bitcoin Close Price (01/01/2019 - 01/04/2019)')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
