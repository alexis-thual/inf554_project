import importlib

import Models.MaxModel

class Assignment:
    def __init__(self, assignment_name, model_name='MaxModel', data=None, verbose=False):
        self.assignment_name = assignment_name

        # Here one loads a module which has the same name as the class
        # one will instantiate right after for the assignment's model.
        module = importlib.import_module("Models." + model_name)
        self.model = getattr(module, model_name)()
        self.model.set_data(data, verbose=verbose)

    def __str__(self):
        return "Assignment with model : " + str(self.assignment_name)

    def __repr__(self):
        return "Assignment with model : " + str(self.assignment_name)
