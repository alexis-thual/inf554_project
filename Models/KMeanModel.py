import pandas as pd
from datetime import timedelta
from math import exp, log, ceil

from .Model import Model

class KMeanModel(Model):
    def __init__(self):
        self.data = pd.DataFrame()
        self.coeff = 1

    def train(self, verbose=False):
        pass

    def index_array(self, index, days_before, days_after, hours_before, hours_after):
        indices = []
        weights = []
        data_len = len(self.data)

        # for days, md in [(days_before, -1), (days_after, 1)]:
        #     for d in range(days):
        #         for hours, mh in [(hours_before, -1), (hours_after, 1)]:
        #             for h in range(0 if mh == -1 else 1, 2 * hours):
        #                 new_index = index + (days + md * d) * 48 * 7 - (2 * hours + mh * h)
        #                 if new_index < data_len:
        #                     result.append(int(new_index))

        for d in range(days_before):
            for h in range(2 * hours_before):
                new_index = index - (days_before - d) * 48 * 7 - (2 *  hours_before - h)
                indices.append(int(new_index))
                weights.append(1 / (2**(d+1)))
            for h in range(2 * hours_after):
                new_index = index - (days_before - d) * 48 * 7 + (h + 1)
                indices.append(int(new_index))
                weights.append(1 / (2**(d+1)))

        # for d in range(days_after):
        #     for h in range(2 * hours_before):
        #         new_index = index + (d + 1) * 48 * 7 - (2 * hours_before - h)
        #         if new_index < data_len:
        #             result.append(int(new_index))
        #     for h in range(2 * hours_after):
        #         new_index = index + (d + 1) * 48 * 7 + (h + 1)
        #         if new_index < data_len:
        #             result.append(int(new_index))

        return weights, indices

    def get_index(self, date):
        return int((date - self.data.loc[0].DATE).total_seconds() / 1800)

    def predict(self, submission=None, verbose=False):
        m = []
        alpha = 0.1
        new_index = self.get_index(pd.to_datetime(submission.DATE))
        weights, indices = self.index_array(new_index, 3, 0, 1, 1)
        values = self.data.loc[indices]

        j = 0
        for i, r in values.iterrows():
            if not pd.isnull(r.CSPL_RECEIVED_CALLS):
                # m = max(m, int(r.CSPL_RECEIVED_CALLS))
                m.append(weights[j] * exp(alpha * r.CSPL_RECEIVED_CALLS))
            else:
                weights[j] = 0
            j += 1

        return ceil(self.coeff * (1/alpha * log(1/sum(weights) * sum(m))))

    def set_params(self, params=None, verbose=False):
        if not params == None:
            self.coeff = float(params)

    def __str__(self):
        return "KMeanModel"

    def __repr__(self):
        return "KMeanModel"
