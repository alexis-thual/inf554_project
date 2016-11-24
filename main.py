import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from assignment import Assignment

# Don't forget to execute the load script before running main.py
# exec('load_data.py'.read())
# Otherwise, the following variables won't be defined.

# Debug mode, prints intermediate values.
verb = True

names = [
    'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Assurances',
    'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Médical', 'Nuit', 'RENAULT',
    'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie',
    'Prestataires', 'Gestion DZ', 'Manager', 'Tech. Total', 'Gestion Clients', 'Mécanicien', 'RTC', 'CAT'
]

def initialize_assignment(name):
    return Assignment(name, model_name='MaxModel', data=df[df.ASS_ASSIGNMENT == name], verbose=verb)

assignments = dict(zip(names, list(map(initialize_assignment, names))))
print("assignments calculated")

for i, row in dfsub[:5].iterrows():
    ass_name = row['ASS_ASSIGNMENT']
    prediction = assignments[ass_name].model.predict(submission=row, verbose=verb)
    dfsub.set_value(i, 'prediction', prediction)

dfsub.to_csv(r'submission.txt', index=None, sep='\t')
