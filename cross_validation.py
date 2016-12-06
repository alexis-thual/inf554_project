import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from slugify import slugify

from assignment import Assignment

verb = False

attribution = {
    'MaxModel': [
        'CMS', 'Gestion', 'Prestataires', 'Gestion DZ',
        'Manager', 'Gestion Clients', 'Mécanicien',
        'Crises', 'Gestion Renault'
    ],
    'KMeanModel': [
        'Domicile', 'Gestion - Accueil Telephonique', 'Médical',
        'RENAULT', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter',
        'Tech. Total', 'CAT', 'Gestion Assurances',
        'Gestion Relation Clienteles', 'Japon', 'Nuit',
        'Regulation Medicale', 'RTC'
    ]
}

def initialize_assignment(ass_name, model_name='MaxModel', params=None, data=None):
    return Assignment(ass_name, model_name=model_name, params=params,
                      data=data, verbose=verb)

def cost_function(test, assignment):
    s = 0
    for i, row in test.iterrows():
        prediction = assignment.model.predict(submission=row)

        # print("row.CSPL_RECEIVED_CALLS : " + str(row.CSPL_RECEIVED_CALLS))
        # print("prediction : " + str(prediction))

        if not pd.isnull(row.CSPL_RECEIVED_CALLS):
            tmp = 0.1 * (row.CSPL_RECEIVED_CALLS - prediction)
            s += np.exp(tmp) - tmp - 1
    return s

def cross_validation():
    params_list = np.arange(1, 1.5, 0.05)
    errors_array = []
    data_ass = dict()
    data_test_ass = dict()
    assignments = dict()

    # Data Loading
    print("Loading Data ...")
    for model_name in attribution:
        for ass_name in attribution[model_name]:
            data_loaded = pd.read_csv('new_data/' + slugify(ass_name) + '.txt', sep='\t', parse_dates=[0])
            data_loaded[['CSPL_RECEIVED_CALLS']] = data_loaded[['CSPL_RECEIVED_CALLS']].apply(pd.to_numeric)

            data_ass[ass_name] = data_loaded

            new_ass = initialize_assignment(ass_name, model_name=model_name, data=data_ass[ass_name])

            new_ass.model.train()

            assignments[ass_name] = new_ass

            X_train, X_test = train_test_split(data_ass[ass_name][40000:], test_size=0.02, random_state=None)

            data_test_ass[ass_name] = X_test

    # Cross Validation algorithm
    print("Cross Validation ...")
    for p in tqdm(params_list):
        error_ass = []

        for model_name in attribution:
            for ass_name in tqdm(attribution[model_name]):
                assignments[ass_name].model.set_params(p)

                error_ass.append((ass_name, cost_function(data_test_ass[ass_name], assignments[ass_name])))

        errors_array.append((p, error_ass))

    print("\n")
    for p, e in errors_array:
        print("--- " + str(p) + " ---")
        print(sum([pair[1] for pair in e]))
        # print(e)

cross_validation()
