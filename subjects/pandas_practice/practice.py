import numpy as np
import pandas as pd
import os
import sys
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def clear_screen():
    key = input('Press enter to continue or x to exit: ')
    if key == 'x':
        sys.exit()

    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def do_plot_and_close(manual_close=False):
    plt.show(block=manual_close)
    if not manual_close:
        plt.pause(5)
        plt.close()

def menu_selection():
    menu = input('Select menu to execute: pk for pokemon; te for temperature: ')
    menu_dict = {
        "pk": menu == 'pk',
        "te": menu == 'te'
    }

    return menu_dict

class ErrorHandler():
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exec_val, exec_traceback):
        if exec_type is not None:
            print(f'Error: {exec_type} {exec_val}')
        
        # If return True, it will surpress the error (not re-raise)
        # If return False, it will re-raise the error
        return True

class TimeExecution():
    def __enter__(self):
        self.start_time = time.time()

        return self
    
    def __exit__(self, exec_type, exec_val, exec_traceback):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

def main():
    menu_dict = menu_selection()
    if menu_dict.get('pk'):
        DATASET_PATH = os.path.join('Pokemon.csv')
        df = pd.read_csv(DATASET_PATH)
        print(df.columns)
        print('-----no loc----')
        print(df[0:5]['Total'])
        print('-----with loc---')
        print(df.loc[[0, 1], ['Total']])
        print('---with multiple loc')
        print(df.loc[df['Total'] > 100].loc[df.HP <= 60])
        print(df.loc[df['Total'] > 100, ['Type 1', 'HP']].loc[df.HP <= 60])
        print('-----with iloc---')
        print(df.iloc[[0]])
        clear_screen()

        print('---sort---')
        print(df.sort_values(['Type 1', 'HP'], ascending=[1, 0]).head(5))
        print('---filter---')
        new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]
        print(new_df)
        clear_screen()

        print('---reset index not drop old index column---')
        new_df = new_df.reset_index()
        print(new_df)
        print('---reset index and drop old index column---')
        new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]
        new_df = new_df.reset_index(drop=True)
        print(new_df)
        print('---reset index and drop old index column and inplace (replace itself)---')
        new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]
        new_df.reset_index(drop=True, inplace=True)
        print(new_df)
        clear_screen()

        print('---groupby---')
        with ErrorHandler():
            print(df.groupby(['Type 1']).mean())
        
        with ErrorHandler():
            print('--using mean()')
            print(df.groupby(['Type 1'])[['Attack', 'Defense']].mean())
            print('--using np.mean with axis')
            print(df.groupby(['Type 1'])[['Attack', 'Defense']].apply(lambda x: np.mean(x, axis=0)))
        clear_screen()

        print('---groupby with sort---')
        with TimeExecution():
            with ErrorHandler():
                print(df.groupby(['Type 1'])[['Attack', 'Defense']].mean().sort_values('Defense', ascending=False))
                print(df.groupby(['Type 1'])[['Attack', 'Defense']].mean().sort_values('Defense', ascending=0))
                print(df.groupby(['Type 1'])[['Attack', 'Defense']].mean().sort_values('Defense', ascending=[0]))
    
    if menu_dict.get('te'):
        print('----Pandas Series action---')
        print('--Base info')
        TEMPERATURE_DATASET_PATH = 'timeseries_daily-minimum-temperatures.csv'
        temp_df = pd.read_csv(TEMPERATURE_DATASET_PATH)
        print(temp_df.shape)
        print(temp_df.dtypes)
        print('---convert to Series dtype')
        print('---Change index')
        temp_df['Date'] = pd.to_datetime(temp_df['Date'])
        temp_df.set_index('Date', inplace=True)
        print(temp_df.head())
        print('---another to parse date to pandas DatetimeIndex')
        temp2_df = pd.read_csv(TEMPERATURE_DATASET_PATH, index_col=0, parse_dates=True)
        print(temp2_df.dtypes)
        print(temp2_df.head())
        clear_screen()

        print('---Time based indexing')
        print('---Add week day month year')
        print('--directly add')
        temp2_df['Year'] = temp2_df.index.year
        print(temp2_df.head())
        print('---other ways to add more columns to existed DF')
        print('--use assign')
        temp2_df = temp2_df.assign(Month=temp2_df.index.month)
        print(temp2_df.head())
        print('--use loc')
        temp2_df.loc[:, 'Weekday Name'] = temp2_df.index.day_name()
        print(temp2_df.head())
        print('--use insert')
        temp2_df.insert(len(temp2_df.columns) - 1, 'Day', temp2_df.index.day)
        print(temp2_df.head())
        print('--use concat')
        print('--first drop Weekday Name and Day')
        temp2_df.drop('Weekday Name', axis=1, inplace=True)
        temp2_df.drop(columns=['Day'], inplace=True)
        print(temp2_df.head())
        print('--pay attention on index (indices) of each left and right df when concat')
        day_df = pd.DataFrame({"Day": temp2_df.index.day}, index=temp2_df.index)
        day_name_df = pd.DataFrame({"Weekday Name": temp2_df.index.day_name()}, index=temp2_df.index)
        temp2_df = pd.concat([temp2_df, day_df, day_name_df], axis=1)
        print(temp2_df.head())
        print(temp2_df['1990-01-01':'1990-01-10'])
        clear_screen()

        print('--convert number to numeric')
        temp2_df['Daily minimum temperatures'] = pd.to_numeric(temp2_df['Daily minimum temperatures'], errors='coerce')
        print(temp2_df.dtypes)
        print('--visualize time-series data')
        sns.set_theme(rc={'figure.figsize': (9, 4)})
        col_to_plot = 'Daily minimum temperatures'
        temp2_df[col_to_plot].plot(linewidth=0.8)
        plt.ylabel(col_to_plot)
        do_plot_and_close()

        print('---Plot by another style')
        temp2_df[col_to_plot].plot(
            marker='.',
            alpha=0.5,
            linestyle='None',
            figsize=(10, 4),
            subplots=True
        )
        plt.ylabel(col_to_plot)
        do_plot_and_close()
        
        print('---Seasonality Visualization')
        col_to_plot = 'Daily minimum temperatures'
        plt.figure(figsize=(10, 4))
        sns.boxplot(data=temp2_df, x='Month', y=col_to_plot)
        plt.ylabel('Temperature')
        plt.title(col_to_plot)
        do_plot_and_close()

        print("---Frequencies")
        print('-- create a date range with frequency')
        print(pd.date_range('1998-03-10', '1998-03-15', freq='D'))
        print(pd.date_range('2004-09-20', periods=10, freq='H'))
        print('--get a subset or a sample of rows')
        time_samples = pd.to_datetime(['1981-01-01', '1981-01-04', '1981-01-08'])
        consum_sample = temp2_df.loc[time_samples, [col_to_plot]].copy()
        print(consum_sample)
        # the above sample is not continuos on datetime index
        # Make it daily frequency without filling the missing data
        print('--Make it daily frequency without filling the missing data')
        consum_freq = consum_sample.asfreq('D')
        print(consum_freq)
        print('--Fill missing data by forward filling way')
        consum_freq['Daily minimum temperatures - Forward Fill'] = consum_sample.asfreq('D', method='ffill')
        print(consum_freq)
        print('--Fill missing data as making the sample frequency')
        new_filled_consum_freq = consum_sample.asfreq('D', method='ffill')
        print(new_filled_consum_freq)

        print('----Resampling')
        data_columns = ['Daily minimum temperatures']
        print(type(temp2_df[data_columns]))
        ts_weekly_mean = temp2_df[data_columns].resample('W').mean()
        print(ts_weekly_mean.head())
        print('--visualize')
        start, end = '1981-01', '1981-12'
        print('--plot daily and weekly resampled time series on the same plot')
        _fig, ax = plt.subplots()
        ax.plot(
            temp2_df.loc[start:end, data_columns],
            marker='.',
            linestyle='-',
            linewidth=0.5,
            label='Daily'
        )
        ax.plot(
            ts_weekly_mean.loc[start:end, data_columns],
            marker='o',
            linestyle='-',
            label='Weekly mean Resample'
        )
        ax.set_ylabel('Temperature')
        ax.legend()

        # custom xticks
        x_ticks = pd.date_range(start, end, freq='2M')
        end_by_month = pd.Period(end).end_time.date() 

        if end_by_month not in x_ticks.to_list():
            x_ticks_lst = x_ticks.to_list()
            x_ticks_lst.append(end_by_month)
            x_ticks = x_ticks_lst

        print(x_ticks)
        plt.xticks(x_ticks, rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        do_plot_and_close()

        ts_annual_df = temp2_df[data_columns].resample('A').sum(min_count=360)
        # The min_count=360 parameter specifies that the sum will only be calculated for a given period
        # if there are at least 360 non-NA/null values in that period.
        # If fewer than 360 valid data points exist, the result for that period will be NaN.
        ts_annual_df = ts_annual_df.set_index(ts_annual_df.index.year)
        ts_annual_df.index.name = 'Year'
        print(ts_annual_df.tail(5))

        ts_annual_df.loc[1981:, data_columns].plot.bar(color='C0', legend=False)
        plt.ylabel("Fraction")
        plt.title('Anual Temperature')
        plt.xticks(rotation=0)
        do_plot_and_close(manual_close=True)

if __name__ == "__main__":
    main()
