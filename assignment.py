import importlib
import unicodedata
from slugify import slugify

class Assignment:
    def __init__(self, assignment_name, model_name='MaxModel', params=None, data=None, verbose=False):
        self.assignment_name = assignment_name

        # Here one loads a module which has the same name as the class
        # one will instantiate right after for the assignment's model.
        module = importlib.import_module("Models." + model_name)
        self.model = getattr(module, model_name)()
        self.model_name = model_name
        self.model.set_data(data, slugify(self.assignment_name), verbose=verbose)
        self.model.set_params(params=params, verbose=verbose)

    def __str__(self):
        return "Assignment with model : " + str(self.model_name)

    def __repr__(self):
        return "Assignment with model : " + str(self.model_name)
