# rm.dropna(inplace=True)

for f in range(30, 36):
    decomposition = seasonal_decompose(rm, freq=7*f)
    decomposition.plot()
    plt.title("Frequence : " + str(f))
    plt.show()

# 31, 32 c'est les meilleurs
decomposition = seasonal_decompose(rm, freq=7*31)
# decomposition.plot()
# plt.title("Frequence : " + str(f))
# plt.show()

observed = decomposition.observed
trend = decomposition.trend
seasonal = decomposition.seasonal
resid = decomposition.resid

x = []
y = []
begin_date = pd.Timestamp(2011, 1, 1)
for d, c in trend.iteritems():
    if not pd.isnull(c):
        x.append((d - begin_date).days)
        y.append(c)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(np.vstack(x), np.vstack(y))

new_seasonal = []
for d, c in seasonal.iteritems():
    days_bet = (d - begin_date).days
    if days_bet in x:
        if not pd.isnull(c):
            new_seasonal.append(c)

new_observed = []
for d, c in observed.iteritems():
    days_bet = (d - begin_date).days
    if days_bet in x:
        if not pd.isnull(c):
            new_observed.append(c)

# plt.plot(x, model.predict(np.vstack(x)))
# plt.plot(x, trend)
# plt.show()

plt.plot(x, model.predict(np.vstack(x)) + np.vstack(new_seasonal))
plt.plot(x, new_observed)
plt.show()

# def prediction(date):
