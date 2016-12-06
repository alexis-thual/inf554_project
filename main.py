import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from slugify import slugify

from assignment import Assignment

# Don't forget to execute the load script before running main.py
# exec('load_data.py'.read())
# Otherwise, the following variables won't be defined.

# Debug mode, prints intermediate values.
verb = False

attribution = {
    'MaxModel': [
        'CMS', 'Gestion', 'Prestataires', 'Gestion DZ',
        'Manager', 'Gestion Clients', 'Mécanicien',
        'Crises', 'Gestion Renault'
    ],
    'Arima': [
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

assignments = dict()

print("Initializing assignments :")

for model_name in attribution:
    for ass_name in tqdm(attribution[model_name]):
        data_ass = pd.read_csv(
            'new_data/' + slugify(ass_name) + '.txt',
            sep='\t',
            usecols=['DATE','CSPL_RECEIVED_CALLS'],
            parse_dates=[0]
        )
        data_ass[['CSPL_RECEIVED_CALLS']] = data_ass[['CSPL_RECEIVED_CALLS']].apply(pd.to_numeric)

        new_ass = initialize_assignment(ass_name, model_name=model_name, data=data_ass)
        new_ass.model.train(verbose=verb)

        assignments[ass_name] = new_ass

print("Calculating predictions :")

for i, row in tqdm(dfsub.iterrows()):
    ass_name = row['ASS_ASSIGNMENT']
    prediction = assignments[ass_name].model.predict(submission=row, verbose=verb)
    dfsub.set_value(i, 'prediction', prediction)

dfsub.to_csv(r'submission.txt', index=None, sep='\t')
