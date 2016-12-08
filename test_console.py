import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tqdm import tqdm
from functools import reduce
from statsmodels.tsa.seasonal import seasonal_decompose


data_ass = pd.read_csv('new_data/japon.txt',sep='\t',parse_dates=[0],index_col=0)
data_ass.dropna(inplace=True)
# data_ass.interpolate(inplace=True)

# Répartition des data par jour de la semaine
days = []
for d in range(7):
    sub_data = data_ass[data_ass.index.weekday == d]
    days.append(sub_data)

# Moyenne par heure et par jour de la semaine
means = []
standard_deviations = []
max_mean = []
max_std = []
for y in range(7):
    data = data_ass[data_ass.index.weekday == y]
    hours_mean = []
    hours_std = []
    for i in range(48):
        sub_array = data.between_time(start_time=str(datetime.timedelta(minutes=i*30)), end_time=str(datetime.timedelta(minutes=((i+1)%48)*30)))
        hours_mean.append((sub_array.mean()).CSPL_RECEIVED_CALLS)
        hours_std.append((sub_array.std()).CSPL_RECEIVED_CALLS)

    means.append(hours_mean)
    standard_deviations.append(hours_std)

    max_mean.append(max(hours_mean))
    max_std.append(max(hours_std))

# Graphique des moyennes et erreurs par jour de la semaine
# for k in range(7):
#     plt.subplot(int("1" + str(7) + str(k+1)))
#     plt.plot(means[k], label=k)
#     plt.errorbar(range(48), means[k], standard_deviations[k][1])
# plt.show()
# print(max_mean / max_mean[0])

# Calcul du coefficient qui minimise le carré de la distance
# au modèle moyen calculé précédement
def beta(X, V):
    if not len(X) == len(V):
        return None
    a = sum(list(map(lambda x: x[0] * x[1], zip(V,V))))
    b = sum(list(map(lambda x: x[0] * x[1], zip(V,X))))
    return b / a

group = data_ass.groupby([data_ass.index.year, data_ass.index.month, data_ass.index.day])

# Calcul sur l'année du coefficient de dilatation donné par beta()
# coeff_list_mon = []
# for i, r in group:
#     date = pd.to_datetime(reduce(lambda a,b: a + str(b), tuple(map(lambda x: str(x) if not len(str(x)) == 1 else ("0" + str(x)), i)), ""), format='%Y%m%d', errors='ignore')
#     wd = date.dayofweek
#     if wd == 0:
#         nl = list(map(lambda x: x[0], r.values.tolist()))
#         l = beta(nl, means[wd])
#         if not l == None:
#             if not l < 0.5:
#                 coeff_list_mon.append((date, l))
#
# coeff_list_tue = []
# for i, r in group:
#     date = pd.to_datetime(reduce(lambda a,b: a + str(b), tuple(map(lambda x: str(x) if not len(str(x)) == 1 else ("0" + str(x)), i)), ""), format='%Y%m%d', errors='ignore')
#     wd = date.dayofweek
#     if wd == 1:
#         nl = list(map(lambda x: x[0], r.values.tolist()))
#         l = beta(nl, means[wd])
#         if not l == None:
#             if not l < 0.5:
#                 coeff_list_tue.append((date, l))

coeff_list = []
for i, r in group:
    date = pd.to_datetime(reduce(lambda a,b: a + str(b), tuple(map(lambda x: str(x) if not len(str(x)) == 1 else ("0" + str(x)), i)), ""), format='%Y%m%d', errors='ignore')
    wd = date.dayofweek
    if wd < 5:
        nl = list(map(lambda x: x[0], r.values.tolist()))
        l = beta(nl, means[wd])
        if not l == None:
            if not l < 0.5:
                coeff_list.append((date, l))

df_coeff = pd.DataFrame(coeff_list)
df_coeff.set_index(0, inplace=True)
# plt.plot(df_coeff)
# plt.show()

# df_coeff_mon = pd.DataFrame(coeff_list_mon)
# df_coeff_mon.set_index(0, inplace=True)
# df_coeff_tue = pd.DataFrame(coeff_list_tue)
# df_coeff_tue.set_index(0, inplace=True)
# plt.plot(df_coeff_mon)
# plt.plot(df_coeff_tue)
# plt.show()

# df_coeff = pd.DataFrame(coeff_list)
# df_coeff.set_index(0, inplace=True)
# df_coeff.dropna(inplace=True)
# df_log = np.log(df_coeff)
# df_log_diff = df_log - df_log.shift()

# Calcul de la trend sur les données extraites
# decomposition = seasonal_decompose(df_coeff, freq=7*10)
# decomposition.plot()
# plt.show()

rm = pd.rolling_mean(df_coeff, center=False, window=7*3)
rm = rm[1]
rm.dropna(inplace=True)

# from sklearn.linear_model import LinearRegression
#
# for deg in range(20, 35):
#     xc = []
#     yc = []
#     begin_date = pd.Timestamp(2011, 1, 1)
#     for d, c in rm.iteritems():
#         if not pd.isnull(c):
#             xc.append((d - begin_date).days)
#             yc.append(c)
#     X = []
#     for i in range(1, deg):
#         X.append(list(map(lambda a: a**i, xc)))
#     datalolz = np.vstack(list(zip(*X)))
#     # yc = np.vstack(yc)
#     model = LinearRegression()
#     model.fit(datalolz, yc)
# #     plt.plot(xc, model.predict(datalolz))
# # plt.show()
#
# xc = np.array(xc)
# yc = np.array(yc)

# plt.plot(x,y)
# plt.show()

# popofit = np.polyfit(x, y, 5)
# plt.plot(popofit)
# plt.show()
