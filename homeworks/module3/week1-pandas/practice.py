import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def main():
    dataset_path = os.path.join('dataset', 'IMDB-Movie-Data.csv')
    df = pd.read_csv(dataset_path, index_col='Title')
    print(df.head())
    print(df.info())
    print(df.describe())
    genre = df['Genre']
    print(genre.head())
    print(df.isnull().sum())

    power_consumption_path = os.path.join('dataset', 'opsd_germany_daily.csv')
    opsd_df = pd.read_csv(power_consumption_path, index_col=0, parse_dates=True)
    opsd_df['Month'] = opsd_df.index.month
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

    plt.show()

if __name__ == "__main__":
    main()
