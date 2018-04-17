from time import time
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import matplotlib.ticker
import sklearn.manifold
import sklearn.datasets

# Next line to silence pyflakes. This import is needed.
mpl_toolkits.mplot3d.Axes3D

n_points = 1000
X, color = sklearn.datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

fig = matplotlib.pyplot.figure(figsize=(15, 8))
matplotlib.pyplot.suptitle("Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14)

ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=matplotlib.pyplot.cm.Spectral)
ax.view_init(4, -72)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    Y = sklearn.manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto', method=method).fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    matplotlib.pyplot.scatter(Y[:, 0], Y[:, 1], c=color, cmap=matplotlib.pyplot.cm.Spectral)
    matplotlib.pyplot.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    matplotlib.pyplot.axis('tight')

t0 = time()
Y = sklearn.manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
matplotlib.pyplot.scatter(Y[:, 0], Y[:, 1], c=color, cmap=matplotlib.pyplot.cm.Spectral)
matplotlib.pyplot.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

t0 = time()
mds = sklearn.manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
matplotlib.pyplot.scatter(Y[:, 0], Y[:, 1], c=color, cmap=matplotlib.pyplot.cm.Spectral)
matplotlib.pyplot.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

t0 = time()
se = sklearn.manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
matplotlib.pyplot.scatter(Y[:, 0], Y[:, 1], c=color, cmap=matplotlib.pyplot.cm.Spectral)
matplotlib.pyplot.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

t0 = time()
tsne = sklearn.manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(2, 5, 10)
matplotlib.pyplot.scatter(Y[:, 0], Y[:, 1], c=color, cmap=matplotlib.pyplot.cm.Spectral)
matplotlib.pyplot.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

matplotlib.pyplot.show()