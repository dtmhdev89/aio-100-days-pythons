import pandas as pd
import os

def main():
    weather_history_path = os.path.join('dataset', 'WeatherHistory2D.csv')
    df = pd.read_csv(weather_history_path, index_col='Formatted Date', parse_dates=True)
    df.index = pd.Index(pd.to_datetime(df.index, utc=True))
    print(df.info())
    print(df.describe())
    print(df.head())
    df.loc[:, 'year'] = df.index.year
    df = df.assign(
        month=df.index.month,
        day=df.index.day,
    )
    print(df.head())

    tweets_path = os.path.join('dataset', 'train.csv')
    tweet_df = pd.read_csv(tweets_path)
    print(tweet_df.columns)
    tweet_df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)
    print(tweet_df.head())

    weather_simple_path = os.path.join('dataset', 'weatherHistory_simple.csv')
    weather_df = pd.read_csv(weather_simple_path)
    print(weather_df.columns)
    grouped_df = weather_df.groupby(['Precip Type']).mean(['Temperature (C)'])
    print(grouped_df)
    weather_df['Temperature (F)'] = weather_df['Temperature (C)'] * (9/5) + 32
    print(weather_df.head(2))
    print(weather_df.iloc[:, [0, 2]])

    fraud_path = os.path.join('dataset', 'creditcard.csv')
    fraud_df = pd.read_csv(fraud_path)
    print(fraud_df.groupby('Class').size())

if __name__ == "__main__":
    main()
