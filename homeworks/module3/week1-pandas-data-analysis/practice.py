import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.dates as mdates

def rating_groups(rating):
    results_np = np.array(['Good', 'Average', 'Bad'])
    cond_arr = [rating >= 7.5, (rating >= 6.0) & (rating < 7.5), rating < 6.0]

    return results_np[cond_arr][-1]

def main():
    dataset_path = os.path.join('dataset', 'IMDB-Movie-Data.csv')
    df = pd.read_csv(dataset_path, index_col='Title')
    print(df.head())
    print(df.info())
    print(df.describe())
    genre = df['Genre']
    print(genre.head())
    filtered_columns = ['Genre', 'Actors', 'Director', 'Rating']
    filtered_data = df[filtered_columns]
    print(filtered_data.head())
    print('-------conditional filtering')
    cond_filterred_data = df[((df['Year'] >= 2010) & (df['Year'] <= 2015))
        & (df['Rating'] < 6.0)
        & (df['Revenue (Millions)'] > df['Revenue (Millions)'].quantile(0.95))
    ]
    print(cond_filterred_data.head())
    print('-------groupby operations')
    print(df.groupby('Director')[['Rating']].mean().head())
    print('-------sorting operations')
    print(df.groupby('Director')[['Rating']].mean().sort_values(['Rating'], ascending=False).head())
    print('-------view missing values')
    print(df.isnull().sum())
    print('-------dealing with missing values: deleting')
    print('===drop entires columns')
    print(df.drop('Metascore', axis=1).head())
    print('===drop rows with na')
    print(df.dropna())
    print(df.head())
    print('-------dealing with missing values: filling')
    revenue_mean = df['Revenue (Millions)'].mean()
    print('revenue mean: ', revenue_mean)
    nan_revenue_indices = df.loc[df['Revenue (Millions)'].isna(), ['Revenue (Millions)']].index
    print('nan revenue indices: ', nan_revenue_indices)
    df['Revenue (Millions)'].fillna(revenue_mean, inplace=True)
    print(df.loc[nan_revenue_indices, ['Revenue (Millions)']])
    print('===apply function')
    df['Rating Category'] = df['Rating'].apply(rating_groups)
    print(df[['Rating', 'Rating Category']].head())

    power_consumption_path = os.path.join('dataset', 'opsd_germany_daily.csv')
    opsd_df = pd.read_csv(power_consumption_path, index_col=0, parse_dates=True)
    opsd_df['Month'] = opsd_df.index.month
    print(opsd_df.loc['2014-01-20':'2014-01-22'])
    cols_plot = ['Consumption', 'Solar', 'Wind']
    axes = opsd_df[cols_plot].plot(marker='.', alpha=0.5, linestyle=None, figsize=(11, 9), subplots=True)
    for ax in axes:
        ax.set_ylabel('Daily Totals (GWh)')
    
    # Seasonality
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, ax in zip(cols_plot, axes):
        sns.boxplot(data=opsd_df, x='Month', y=name, ax=ax)
        ax.set_ylabel('GWh')
        ax.set_title(name)
        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')

    # Frequencies
    print('-----Frequencies')
    times_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])
    consume_sample = opsd_df.loc[times_sample, ['Consumption']].copy()
    print(consume_sample)
    consume_freq = consume_sample.asfreq(freq='D')
    print('consume frequency: ', consume_freq)
    consume_freq['Consume Freq - Forward Fill'] = consume_sample.asfreq('D', method='ffill')
    print('after forward fill')
    print(consume_freq.head())
    print('------Resampling')
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
    opsd_weekly_mean = opsd_df[data_columns].resample('w').mean()
    print(opsd_weekly_mean.info())
    print('------visualize solar in daily and weekly')
    start, end = '2017-01', '2017-06'
    fig, ax = plt.subplots()
    ax.plot(opsd_df.loc[start:end, 'Solar'], marker='.', linestyle='-', linewidth=0.5, label='Daily')
    ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'], marker='o', markersize=8, linestyle='-', label='Weekly mean Resample')
    ax.set_ylabel('Solar Production (GWh)')
    ax.legend()

    opsd_annual = opsd_df[data_columns].resample('YE').sum(min_count=360)
    opsd_annual.set_index(opsd_annual.index.year, inplace=True)
    opsd_annual.index.name = 'Year'
    opsd_annual['Wind+Solar/Consumption'] = opsd_annual['Wind+Solar'] / opsd_annual['Consumption']
    print(opsd_annual.tail(3))

    print('----Visualize wind and solar contribution')
    axx = opsd_annual.loc[2012:, ['Wind+Solar/Consumption']].plot.bar(color='C0')
    axx.set_ylabel('Fraction')
    axx.set_ylim(0, 0.3)
    axx.set_title('Wind + Solar share of Annual Electricity Consumption')

    print('-----Rolling window')
    opsd_7d = opsd_df[data_columns].rolling(7, center=True).mean()
    print(opsd_7d.head(20))

    print('------trends')
    opsd_365d = opsd_df[data_columns].rolling(window=365, center=True, min_periods=360).mean()
    _fig, ax  = plt.subplots()
    ax.plot(opsd_df['Consumption'], marker='.', markersize=2, color='0.6', linestyle='None', label='Daily')
    ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
    ax.plot(opsd_365d['Consumption'], color='0.2', linewidth=3, label='Trend (365-d Rolling Mean)')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption (GWh)')
    ax.set_title('Trends in Electricity Consumption')

    print('===wind, solar 365d')
    _fig, ax = plt.subplots()
    for feature in ['Wind', 'Solar', 'Wind+Solar']:
        ax.plot(opsd_365d[feature], label=feature)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylim(0, 400)
        ax.legend()
        ax.set_ylabel('Production (GWh)')
        ax.set_title('Trends in  Electricity Production (365-d Rolling Mean)')

    plt.show()

if __name__ == "__main__":
    main()
