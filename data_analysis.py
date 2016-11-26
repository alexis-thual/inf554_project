import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def restr(dataf, assignment, interv=None, day=None):
    # Specify an assignment
    dataf = dataf[dataf.ASS_ASSIGNMENT == assignment]
    dataf.drop('ASS_ASSIGNMENT',1, inplace=True)

    # Group over the different ACD codes
    dataf = dataf.groupby('DATE').sum()
    dataf.sort_index(0, inplace=True)

    # Specify a time interval
    if not interv is None:
        dataf = dataf.between_time(interv, interv)

    # Restrict to specific day (Monday=0, Sunday=6)
    if not day is None:
        dataf = dataf[dataf.index.weekday == wday]

    dataf.dropna(inplace=True)
    # dataf.reset_index(inplace=True)

    return dataf

names = [
    'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique',
    'Gestion Assurances', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon',
    'Médical', 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', 'Services',
    'Tech. Axa', 'Tech. Inter', 'Téléphonie', 'Prestataires', 'Gestion DZ',
    'Manager', 'Tech. Total', 'Gestion Clients', 'Mécanicien', 'RTC', 'CAT'
]

# for assignment in names:
#     for i in range(2):
#         ts = restr(df, assignment)
#         plt.plot(ts)
#         plt.title(assignment)
#     plt.show()

ts = restr(df, 'Téléphonie')
plt.plot(ts)
plt.title('Téléphonie')
plt.show()

enorme_pic_names = [
    'Crises'
]

max_model_names = [
    'CMS', 'Gestion', 'Prestataires', 'Gestion DZ', 'Manager', 'Gestion Clients',
    'Mécanicien'
]

# SAP pourrait peut-être se retrouver dans a_lancienne_petit_names
# Téléphonie : le nombre d'appels explose vers juin 2013

a_lancienne_names = [
    'Domicile', 'Gestion - Accueil Telephonique', 'Médical', 'RENAULT',
    'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 'Tech. Total', 'CAT'
]

# Nuit a une intensité élevée dans les premiers mois, faible par la suite (zone à prévoir)

a_lancienne_petit_names = [
    'Gestion Assurances', 'Gestion Relation Clienteles', 'Japon', 'Nuit',
    'Regulation Medicale', 'RTC'
]

zero_names = [
    'Gestion Renault'
]
