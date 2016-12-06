import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

from .Model import Model

class Arima(Model):
    def __init__(self):
        self.data = pd.DataFrame()

    def train(self, verbose=False):
        # ts_expavg_diff = self.data - moving_expavg
        # ts_shift = self.data - self.data.shift()

        plt.plot(self.data, color='blue')
        plt.show()

    def predict(self, submission=None, verbose=False):
        return 1.4

    def set_params(self, params=None, verbose=False):
        pass

    def __str__(self):
        return "Arima"

    def __repr__(self):
        return "Arima"
