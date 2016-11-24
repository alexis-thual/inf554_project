import pandas as pd

from .Model import Model

class MaxModel(Model):
    def __init__(self):
        self.maximum = None
        self.data = pd.DataFrame()

    def predict(self, submission=None, verbose=False):
        if self.data.empty:
            print("No data was set to this model, therefore prediction can't be made.")
        elif self.maximum == None:
            if verbose:
                print("Maximum calculated for the first time")
            self.maximum = int(self.data.max()['CSPL_RECEIVED_CALLS'])
        return self.maximum

    def __str__(self):
        return "MaxModel : " + str(self.maximum)

    def __repr__(self):
        return "MaxModel : " + str(self.maximum)
