import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from assignment import Assignment

# Don't forget to execute the load script before running main.py
# exec('load_data.py'.read())
# Otherwise, the following variables won't be defined.

# Debug mode, prints intermediate values.
verb = True

attribution = {
    'MaxModel': [
        'CMS', 'Gestion', 'Prestataires', 'Gestion DZ',
        'Manager', 'Gestion Clients', 'Mécanicien',
        'Crises', 'Gestion Renault'
    ],
    'KMeanModel': [
        'Domicile', 'Gestion - Accueil Telephonique', 'Médical',
        'RENAULT', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter',
        'Téléphonie', 'Tech. Total', 'CAT', 'Gestion Assurances',
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
        tmp = -0.1 * (assignments.model.predict(row, verbose=verb) - row.CSPL_RECEIVED_CALLS)
        s += np.exp(tmp) - tmp - 1
    return s

def cross_validation(data):
    params_list = [1]
    for p in params_list:
        assignments = dict()

        for model_name in attribution:
            for ass_name in attribution[model_name]:
                X_train, X_test = train_test_split(
                    df[df.ASS_ASSIGNMENT == ass_name], test_size=0.1, random_state=None)
                new_ass = initialize_assignment(ass_name, model_name, p, X_train)
                new_ass.model.train(verbose=verb)

                total_error = cost_function(X_test, new_ass)

                assignments[ass_name] = new_ass

        print("Assignments calculated.")

assignments = dict()

for model_name in attribution:
    for ass_name in attribution[model_name]:
        new_ass = initialize_assignment(ass_name, model_name=model_name, data=df[df.ASS_ASSIGNMENT == ass_name])
        new_ass.model.train(verbose=verb)
        assignments[ass_name] = new_ass

print("Assignments calculated.")


for i, row in dfsub[:5].iterrows():
    ass_name = row['ASS_ASSIGNMENT']
    prediction = assignments[ass_name].model.predict(submission=row, verbose=verb)
    dfsub.set_value(i, 'prediction', prediction)

dfsub.to_csv(r'submission.txt', index=None, sep='\t')
