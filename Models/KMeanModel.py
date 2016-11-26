import pandas as pd
from datetime import timedelta

from .Model import Model

class KMeanModel(Model):
    def __init__(self):
        self.data = pd.DataFrame()

    def train(self, verbose=False):
        if verbose:
            print("KMeanModel trained.")
        pass

    def predict(self, submission=None, verbose=False):
        if self.data.empty:
            print("No data was set.")
        else:
            date_next = submission.DATE + timedelta(days=7)
            date_previous = submission.DATE + timedelta(days=-7)
            means = []

            for date in [date_next, date_previous]:
                m = 0
                c = 0
                values = self.data[str(date + timedelta(seconds=-3600)):str(date + timedelta(seconds=3600))]
                for i, r in values.iterrows():
                    m += int(r.CSPL_RECEIVED_CALLS)
                    c += 1
                if c > 0:
                    m /= c
                    means.append(m)

            return sum(means) / len(means)

    def set_params(self, params=None, verbose=False):
        pass

    def __str__(self):
        return "KMeanModel"

    def __repr__(self):
        return "KMeanModel"
