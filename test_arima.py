import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from slugify import slugify
from statsmodels.tsa.stattools import adfuller
from assignment import Assignment

verb = False

def initialize_assignment(ass_name, model_name='MaxModel', params=None, data=None):
    return Assignment(ass_name, model_name=model_name, params=params,
                      data=data, verbose=verb)

def test_stationarity(timeseries, w):
    rolmean = pd.rolling_mean(timeseries, window=w)
    rolstd = pd.rolling_std(timeseries, window=w)
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

data_ass = pd.read_csv(
    'new_data/tech-axa.txt',
    sep='\t',
    usecols=['DATE','CSPL_RECEIVED_CALLS'],
    parse_dates=[0]
)
data_ass[['CSPL_RECEIVED_CALLS']] = data_ass[['CSPL_RECEIVED_CALLS']].apply(pd.to_numeric)
data_ass.set_index('DATE', inplace=True)

test_stationarity(data_ass)

w = 1000

data_ass = pd.read_csv(
    'new_data/telephonie.txt',
    sep='\t',
    parse_dates=[0],
    index_col=0
)
data_ass.dropna(inplace=True)

ts = data_ass['CSPL_RECEIVED_CALLS']
test_stationarity(ts, w)

ts_log = np.log(ts+1)
plt.plot(ts_log)

moving_avg = pd.Series.rolling(ts_log, center=False, window=w).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff, w)

expwighted_avg = pd.ewma(ts_log, halflife=w)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
plt.show()

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff, w)

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

# new_ass = initialize_assignment('Tech. Axa', model_name='Arima', data=data_ass)
# new_ass.model.train(verbose=verb)
