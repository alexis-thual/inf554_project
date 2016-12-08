# exec(open('test_console.py').read())

from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
from statsmodels.graphics.utils import _import_mpl
from statsmodels.compat.python import iteritems
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.compat.python import lmap

def seasonal_mean(x, freq):
    return np.array([pd_nanmean(x[i::freq]) for i in range(freq)])

def seasonal_decompose(x, freq=None):
    _pandas_wrapper, pfreq = _maybe_get_pandas_wrapper_freq(x)
    x = np.asanyarray(x).squeeze()
    nobs = len(x)

    if freq % 2 == 0:  # split weights at ends
        filt = np.array([.5] + [1] * (freq - 1) + [.5]) / freq
    else:
        filt = np.repeat(1./freq, freq)

    trend = convolution_filter(x, filt=filt, nsides=1)
    # trend = convolution_filter(x, filt=filt, nsides=2)
    detrended = x - trend

    period_averages = seasonal_mean(detrended, freq)
    period_averages -= np.mean(period_averages)
    seasonal = np.tile(period_averages, nobs // freq + 1)[:nobs]

    # new_frew = 7 * 3
    # new_curve = detrended - seasonal
    # new_period_averages = seasonal_mean(new_curve, new_frew)
    # new_period_averages -= np.mean(new_period_averages)
    # new_seasonal = np.tile(new_period_averages, nobs // new_frew + 1)[:nobs]

    # resid = new_curve - new_seasonal
    resid = detrended - seasonal

    # results = lmap(_pandas_wrapper, [x, trend, seasonal, new_seasonal, resid])
    # return DecomposeResult(observed=results[0], trend=results[1], seasonal=results[2], new_seasonal=results[3], resid=results[4])
    results = lmap(_pandas_wrapper, [x, trend, seasonal, resid])
    return DecomposeResult(observed=results[0], trend=results[1], seasonal=results[2], resid=results[3])

class DecomposeResult(object):
    def __init__(self, **kwargs):
        for key, value in iteritems(kwargs):
            setattr(self, key, value)
        self.nobs = len(self.observed)

    def plot(self, color='blue'):
        plt = _import_mpl()
        # fig, axes = plt.subplots(5, 1, sharex=True)
        fig, axes = plt.subplots(4, 1, sharex=True)
        if hasattr(self.observed, 'plot'):  # got pandas use it
            self.observed.plot(ax=axes[0], legend=False, color=color)
            axes[0].set_ylabel('Observed')
            self.trend.plot(ax=axes[1], legend=False, color=color)
            axes[1].set_ylabel('Trend')
            self.seasonal.plot(ax=axes[2], legend=False, color=color)
            axes[2].set_ylabel('Seasonal')
            # self.new_seasonal.plot(ax=axes[3], legend=False, color=color)
            # axes[3].set_ylabel('New Seasonal')
            # self.resid.plot(ax=axes[4], legend=False, color=color)
            # axes[4].set_ylabel('Residual')
            self.resid.plot(ax=axes[3], legend=False, color=color)
            axes[3].set_ylabel('Residual')
        else:
            axes[0].plot(self.observed, color=color)
            axes[0].set_ylabel('Observed')
            axes[1].plot(self.trend, color=color)
            axes[1].set_ylabel('Trend')
            axes[2].plot(self.seasonal, color=color)
            axes[2].set_ylabel('Seasonal')
            # axes[3].plot(self.new_seasonal, color=color)
            # axes[3].set_ylabel('New Seasonal')
            # axes[4].plot(self.resid, color=color)
            # axes[4].set_ylabel('Residual')
            # axes[4].set_xlabel('Time')
            # axes[4].set_xlim(0, self.nobs)
            axes[3].plot(self.resid, color=color)
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Time')
            axes[3].set_xlim(0, self.nobs)

        fig.tight_layout()
        return fig

freq = 7 * 4
decomposition = seasonal_decompose(df_coeff, freq=freq)
decomposition.plot(color="seagreen")
plt.show()
