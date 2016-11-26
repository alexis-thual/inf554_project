import pandas as pd

from .Model import Model

class MaxModel(Model):
    def __init__(self):
        self.maximum = None
        self.data = pd.DataFrame()

    def train(self, verbose=False):
        if verbose:
            print("MaxModel trained.")
        self.maximum = int(self.data.max()['CSPL_RECEIVED_CALLS'])

    def predict(self, submission=None, verbose=False):
        if self.data.empty:
            print("No data was set.")
        elif self.maximum == None:
            print("Training has to be performed.")
        return self.maximum

    def set_params(self, params=None, verbose=False):
        pass

    def __str__(self):
        return "MaxModel : " + str(self.maximum)

    def __repr__(self):
        return "MaxModel : " + str(self.maximum)
