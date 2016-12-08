# plt.plot(df_coeff)
# plt.show()

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


kde_funcs = [kde_statsmodels_u, kde_statsmodels_m, kde_scipy, kde_sklearn]
kde_funcnames = ['Statsmodels-U', 'Statsmodels-M', 'Scipy', 'Scikit-learn']

# print("Package Versions:")
# import sklearn; print("  scikit-learn:", sklearn.__version__)
# import scipy; print("  scipy:", scipy.__version__)
# import statsmodels; print("  statsmodels:", statsmodels.__version__)

from scipy.stats.distributions import norm

# The grid we'll use for plotting
# x_grid = np.linspace(-4.5, 3.5, 1000)
x_grid = xc

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
# x = np.concatenate([norm(-1, 1.).rvs(400), norm(1, 0.3).rvs(100)])
x = yc
pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))
# pdf_true = yc

# Plot the three kernel density estimates
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(13, 3))
fig.subplots_adjust(wspace=0)

for i in range(4):
    pdf = kde_funcs[i](x, x_grid, bandwidth=0.2)
    ax[i].plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax[i].plot(x_grid, pdf_true, color='gray', alpha=0.5, lw=3)
    # ax[i].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax[i].set_title(kde_funcnames[i])
    # ax[i].set_xlim(-4.5, 3.5)

# from IPython.display import HTML
# HTML("<font color='#666666'>Gray = True underlying distribution</font><br>"
#      "<font color='6666ff'>Blue = KDE model distribution (500 pts)</font>")

fig, ax = plt.subplots()
for bandwidth in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # ax.plot(x_grid, kde_sklearn(x, x_grid, bandwidth=bandwidth),
            # label='bw={0}'.format(bandwidth), linewidth=3, alpha=0.5)
    ax.plot(x_grid, kde_sklearn(x, x_grid, bandwidth=bandwidth))

# ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
# bins = np.arange(0,2)
# ax.hist(x,bins)
# ax.set_xlim(-4.5, 3.5)
ax.plot(x_grid, x)
ax.legend(loc='upper left')

plt.show()
