from pandas import DataFrame
from pandas import concat
import pandas as pd
from numpy import asarray

from dataset_man import dataset_manager

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from matplotlib import pyplot

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def series_to_supervised_v2(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    print('before nan drop:--\n', agg.loc[-12:, :])

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg.values

def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

def main():
    df = DataFrame()
    df['t'] = [x for x in range(10)]
    print(df)

    df['t-1'] = df['t'].shift(1)
    print(df)

    df['t+1'] = df['t'].shift(-1)
    print(df)

    values = [x for x in range(10)]
    data = series_to_supervised(values)
    print(data)

    data = series_to_supervised(values, 2, 2)
    print(data)

    raw = DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values

    print(raw)

    data = series_to_supervised(values)
    print(data)

    # series = read_csv('/content/daily-total-female-births.csv', header=0, index_col=0)
    series = pd.DataFrame(dataset_manager.load_dataset('m3.random.forest.20240911.daily-total-female-births'))
    series['Date'] = pd.to_datetime(series['Date'])
    series.set_index('Date', inplace=True)
    print(series.head())

    monthly_values = series['Births'].resample('M').sum()

    values = series.values
    # plot dataset
    pyplot.plot(values)
    pyplot.show()

    pyplot.plot(monthly_values)
    pyplot.show()

    print('before:--\n', values[-12:])

    data = series_to_supervised_v2(values, n_in=6)
    print('after:--\n', data[-12:])

    # evaluate
    mae, y, yhat = walk_forward_validation(data, 12)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    pyplot.plot(y, label='Expected')
    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()

if __name__ == "__main__":
    main()
