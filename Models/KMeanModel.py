import pandas as pd
from datetime import timedelta

from .Model import Model

class KMeanModel(Model):
    def __init__(self):
        self.data = pd.DataFrame()

    def train(self, verbose=False):
        pass

    def index_array(self, index, days_before, days_after, hours_before, hours_after):
        result = []
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
                result.append(int(new_index))
            for h in range(2 * hours_after):
                new_index = index - (days_before - d) * 48 * 7 + (h + 1)
                result.append(int(new_index))

        for d in range(days_after):
            for h in range(2 * hours_before):
                new_index = index + (d + 1) * 48 * 7 - (2 * hours_before - h)
                if new_index < data_len:
                    result.append(int(new_index))
            for h in range(2 * hours_after):
                new_index = index + (d + 1) * 48 * 7 + (h + 1)
                if new_index < data_len:
                    result.append(int(new_index))

        return result

    def get_index(self, date):
        return int((date - self.data.loc[0].DATE).total_seconds() / 1800)

    def predict(self, submission=None, verbose=False):
        m = 0
        new_index = self.get_index(pd.to_datetime(submission.DATE))
        values = self.data.loc[self.index_array(new_index, 3, 3, 1, 1)]

        for i, r in values.iterrows():
            if not pd.isnull(r.CSPL_RECEIVED_CALLS):
                m = max(m, int(r.CSPL_RECEIVED_CALLS))

        return m

    def bandaid_data(self):
        dates_to_be_added = []
        d2 = self.data.loc[0].DATE

        for i, row in assignments['Domicile'].model.data[1:].iterrows():
            d1 = row.DATE
            e = int((d1-d2).total_seconds())
            if not e == 1800:
                # print("e : " + str(int(e/1800) - 1) + " i : " + str(i))
                for k in range(int(e / 1800) - 1):
                    dates_to_be_added.append(d2 + timedelta(seconds=(1800*(k+1))))
            d2 = d1

        nndf = pd.DataFrame(list(map(lambda d: [d, 0], dates_to_be_added)), columns=['DATE', 'CSPL_RECEIVED_CALLS'])

        self.data = self.data.append(nndf)

        self.data = self.data.groupby('DATE').sum()
        self.data.sort_index(0, inplace=True)
        self.data.reset_index(inplace=True)

    def set_params(self, params=None, verbose=False):
        pass

    def __str__(self):
        return "KMeanModel"

    def __repr__(self):
        return "KMeanModel"
