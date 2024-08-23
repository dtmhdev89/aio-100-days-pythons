import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def main():
    weather1_path = os.path.join('dataset', 'weatherHistory_v1.csv')
    data_pd = pd.read_csv(weather1_path)
    data_column = 'Temperature (C)'
    temperature_small_pd = data_pd[data_column][0:200]

    fig = plt.figure(figsize=(10, 8))
    rows, cols = (2, 1)
    ax = fig.add_subplot(rows, cols, 1)

    # preprocessing data
    ax.plot(
        temperature_small_pd,
        linestyle='-',
        linewidth='0.8'
    )

    # fill nan
    temperature_small_pd = temperature_small_pd.interpolate()
    ax = fig.add_subplot(rows, cols, 2)
    ax.plot(
        temperature_small_pd,
        linestyle='-',
        linewidth='0.8'
    )
    ax.set_title('Interpolate missing data')

    # visualize box plot
    box_plot_fig = plt.figure(figsize=(10, 8))

    ax = box_plot_fig.add_subplot(rows, cols, 1)
    ax.plot(temperature_small_pd, linestyle='-', linewidth='0.8')

    ax = box_plot_fig.add_subplot(rows, cols, 2)
    ax.boxplot(temperature_small_pd)
    ax.set_title('Box Plot')

    # histogram
    _fig, axs = plt.subplots(nrows=2, figsize=(10, 8))

    axs[0].hist(temperature_small_pd, bins=30, density=True, alpha=0.6)
    axs[0].set_title('Histogram')

    # Calculate KDE
    kde = gaussian_kde(temperature_small_pd)
    x_grid = np.linspace(min(temperature_small_pd), max(temperature_small_pd), 200)
    y_kde = kde(x_grid)

    # Plot the KDE
    axs[1].plot(x_grid, y_kde, label='KDE')
    axs[1].set_title('Kernel Density Estimation')

    # SMA - Simple moving average
    sma_data_pd = temperature_small_pd.rolling(window=5).mean()
    _fig, ax = plt.subplots()
    ax.plot(
        temperature_small_pd,
        linestyle='-',
        linewidth='0.8',
        label='Original Data'
    )

    ax.plot(
        sma_data_pd,
        linestyle='-',
        linewidth='1',
        label="SMA"
    )
    ax.set_title('Noise Reduction')

    # EMS - Exponential moving average | EWA - Exponential Weighted Average | EWM
    ewm_data_pd = temperature_small_pd.ewm(com=9).mean()

    ax.plot(
        ewm_data_pd,
        linestyle='-',
        linewidth='1.5',
        label='EWM'
    )
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
