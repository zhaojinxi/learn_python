import numpy
import matplotlib.pyplot
import scipy
import sklearn.neighbors

#----------------------------------------------------------------------
# Plot the progression of histograms to kernels
numpy.random.seed(1)
N = 20
X = numpy.concatenate((numpy.random.normal(0, 1, int(0.3 * N)), numpy.random.normal(5, 1, int(0.7 * N))))[:, numpy.newaxis]
X_plot = numpy.linspace(-5, 10, 1000)[:, numpy.newaxis]
bins = numpy.linspace(-5, 10, 10)

fig, ax = matplotlib.pyplot.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# histogram 1
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# histogram 2
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# tophat KDE
kde = sklearn.neighbors.KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 0].fill(X_plot[:, 0], numpy.exp(log_dens), fc='#AAAAFF')
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian KDE
kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], numpy.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

for axi in ax.ravel():
    axi.plot(X[:, 0], numpy.zeros(X.shape[0]) - 0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')

for axi in ax[1, :]:
    axi.set_xlabel('x')

#----------------------------------------------------------------------
# Plot all available kernels
X_plot = numpy.linspace(-6, 6, 1000)[:, None]
X_src = numpy.zeros((1, 1))

fig, ax = matplotlib.pyplot.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)

def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x

for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']):
    axi = ax.ravel()[i]
    log_dens = sklearn.neighbors.KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
    axi.fill(X_plot[:, 0], numpy.exp(log_dens), '-k', fc='#AAAAFF')
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(matplotlib.pyplot.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(matplotlib.pyplot.MultipleLocator(1))
    axi.yaxis.set_major_locator(matplotlib.pyplot.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title('Available Kernels')

#----------------------------------------------------------------------
# Plot a 1D density example
N = 100
numpy.random.seed(1)
X = numpy.concatenate((numpy.random.normal(0, 1, int(0.3 * N)), numpy.random.normal(5, 1, int(0.7 * N))))[:, numpy.newaxis]

X_plot = numpy.linspace(-5, 10, 1000)[:, numpy.newaxis]

true_dens = (0.3 * scipy.stats.norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * scipy.stats.norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = matplotlib.pyplot.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2, label='input distribution')

for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    kde = sklearn.neighbors.KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], numpy.exp(log_dens), '-', label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * numpy.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
matplotlib.pyplot.show()