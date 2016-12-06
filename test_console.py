import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from slugify import slugify
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.filtertools import convolution_filter
from pandas.core.nanops import nanmean as pd_nanmean

def seasonal_mean(x, freq):
    return np.array([pd_nanmean(x[i::freq]) for i in range(freq)])

def seasonal_decompose(x, freq=None):
    freq_trend = 48 * 7 * 3
    freq_week = 48 * 7 * 3
    freq_day = 48 * 7 * 3

    if not freq == None:
        freq_trend = freq

    trend = 0
    seasonal_week = 0
    seasonal_day = 0
    resid = 0

    x = np.asanyarray(x).squeeze()
    nobs = len(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("This function does not handle missing values")

    filt = np.repeat(1./freq_trend, freq_trend)
    trend = convolution_filter(x, filt)

    # detrended = x - trend
    #
    # period_averages_week = seasonal_mean(detrended, freq_week)
    # period_averages_week -= np.mean(period_averages_week)
    #
    # seasonal_week = np.tile(period_averages_week, nobs // freq_week + 1)[:nobs]
    #
    # resid_week = detrended - seasonal_week
    #
    # period_averages_day = seasonal_mean(resid_week, freq_day)
    # period_averages_day -= np.mean(period_averages_day)
    #
    # seasonal_day = np.tile(period_averages_day, nobs // freq_day + 1)[:nobs]
    #
    # resid = resid_week - seasonal_day
    #
    return (resid, seasonal_day, seasonal_week, trend)

data_ass = pd.read_csv('new_data/domicile.txt',sep='\t',parse_dates=[0],index_col=0)
data_ass.dropna(inplace=True)
# data_ass.interpolate(inplace=True)

freq_trend = 48 * 7 * 10
filt = np.repeat(1./freq_trend, freq_trend)
trend = convolution_filter(days[0], filt)
plt.subplot(211)
plt.plot(days[0], label="Mondays")
plt.legend(loc='best')
plt.subplot(212)
plt.plot(trend, label="Trend")
plt.legend(loc='best')
plt.show()

hours = dict()
for i in range(48):
    sub_array = days[0].between_time(start_time=str(datetime.timedelta(minutes=i*30)), end_time=str(datetime.timedelta(minutes=((i+1)%48)*30)))
    hours[i] = (sub_array.mean()).CSPL_RECEIVED_CALLS
print(*hours)

total_hours = []
for y in range(3*7):
    data = data_ass[(data_ass.index.year == (2011+(y%3))) & (data_ass.index.weekday == (y//3))]
    hours_mean = []
    hours_std = []
    for i in range(48):
        sub_array = data.between_time(start_time=str(datetime.timedelta(minutes=i*30)), end_time=str(datetime.timedelta(minutes=((i+1)%48)*30)))
        hours_mean.append((sub_array.mean()).CSPL_RECEIVED_CALLS)
        hours_std.append((sub_array.std()).CSPL_RECEIVED_CALLS)
    total_hours.append((hours_mean, hours_std))

x = range(48)
nd = 7
for k in range(nd):
    plt.subplot(int("1" + str(nd) + str(k+1)))
    for y in range(3):
        plt.plot(total_hours[k*3 + y][0], label=k)
        plt.errorbar(x, total_hours[k*3 + y][0], total_hours[k*3 + y][1])
plt.show()

ts = data_ass['CSPL_RECEIVED_CALLS']
ts = np.log(ts + 0.1)
plt.subplot(511)
plt.plot(ts, label='Original')
plt.legend(loc='best')

c = 1
for i in range(5,9):
    decomposition = seasonal_decompose(ts, freq=48*7*i)
    resid = decomposition[0]
    seasonal_day = decomposition[1]
    seasonal_week = decomposition[2]
    trend = decomposition[3]
    plt.subplot(511 + c)
    plt.plot(trend, label='freq:'+str(48*7*i))
    plt.legend(loc='best')
    c += 1

# plt.subplot(512)
# plt.plot(resid, label='resid')
# plt.legend(loc='best')
# plt.subplot(513)
# plt.plot(seasonal_day,label='seasonal_day')
# plt.legend(loc='best')
# plt.subplot(514)
# plt.plot(seasonal_week, label='seasonal_week')
# plt.legend(loc='best')
# plt.subplot(515)
# plt.plot(trend, label='trend')
# plt.legend(loc='best')

plt.tight_layout()
plt.show()

#
# w = 48*7
# def test_stationarity(timeseries, w):
#     rolmean = pd.rolling_mean(timeseries, window=w)
#     rolstd = pd.rolling_std(timeseries, window=w)
#     orig = plt.plot(timeseries, color='blue',label='Original')
#     mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
#     print('Results of Dickey-Fuller Test:')
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
#     print(dfoutput)
# data_ass = pd.read_csv('new_data/domicile.txt',sep='\t',parse_dates=[0],index_col=0)
# data_ass.dropna(inplace=True)
# data_ass.fillna(method='pad', inplace=True)
# ts = data_ass['CSPL_RECEIVED_CALLS']
# ts_log = np.log(ts+0.1)
# decomposition = seasonal_decompose(ts_log, freq=48*7*3)
#
# moving_avg = pd.Series.rolling(ts, center=False, window=w).mean()
# moving_avg = pd.Series.rolling(ts, center=False, window=4).max()
# plt.plot(moving_avg)
# plt.plot(ts)
# plt.show()
#
# ts_log_moving_avg_diff = ts_log - moving_avg
# ts_log_moving_avg_diff.dropna(inplace=True)
# expwighted_avg = pd.ewma(ts_log, halflife=w)
# ts_log_ewma_diff = ts_log - expwighted_avg
# ts_log_diff = ts_log - ts_log.shift()
# ts_log_diff.dropna(inplace=True)
#
# for i, r in data_ass.iterrows():
#     if pd.isnull(r.CSPL_RECEIVED_CALLS):
#         print(i)
#
# ts = data_ass['CSPL_RECEIVED_CALLS']
# ts_log = moving_avg
#
# decomposition = seasonal_decompose(moving_avg, model='additive', freq=48*7*4)
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
# plt.subplot(411)
# plt.plot(moving_avg, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
#
# decomposition2 = seasonal_decompose(residual, freq=48*3)
# trend2 = decomposition2.trend
# seasonal2 = decomposition2.seasonal
# residual2 = decomposition2.resid
# plt.subplot(411)
# plt.plot(residual, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend2, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal2,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual2, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
