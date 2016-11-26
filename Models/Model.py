import pandas as pd

class Model:
    def __init__(self):
        self.data = None

    def preprocess(self, data):
        data.drop('ASS_ASSIGNMENT', 1, inplace=True)
        data = data.groupby('DATE').sum()

        return data
        # data.sort_index(0, inplace=True)
        # data.set_index('DATE', inplace=True)

    def set_data(self, data, verbose=False):
        self.data = self.preprocess(data)

        if verbose:
            print("Data added")
