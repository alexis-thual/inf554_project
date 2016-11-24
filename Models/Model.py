import pandas as pd

class Model:
    def __init__(self):
        self.data = None

    def set_data(self, data, verbose=False):
        if verbose:
            print("Data added")
        self.data = data
