import time
import numpy
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import matplotlib.ticker
import sklearn.manifold
import sklearn.utils

# Next line to silence pyflakes.
mpl_toolkits.mplot3d.Axes3D

# Variables for manifold learning.
n_neighbors = 10
n_samples = 1000

# Create our sphere.
random_state = sklearn.utils.check_random_state(0)
p = random_state.rand(n_samples) * (2 * numpy.pi - 0.55)
t = random_state.rand(n_samples) * numpy.pi

# Sever the poles from the sphere.
indices = ((t < (numpy.pi - (numpy.pi / 8))) & (t > ((numpy.pi / 8))))
colors = p[indices]
x, y, z = numpy.sin(t[indices]) * numpy.cos(p[indices]), numpy.sin(t[indices]) * numpy.sin(p[indices]), numpy.cos(t[indices])

# Plot our dataset.
fig = matplotlib.pyplot.figure(figsize=(15, 8))
matplotlib.pyplot.suptitle("Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14)

ax = fig.add_subplot(251, projection='3d')
ax.scatter(x, y, z, c=p[indices], cmap=matplotlib.pyplot.cm.rainbow)
ax.view_init(40, -10)

sphere_data = numpy.array([x, y, z]).T

# Perform Locally Linear Embedding Manifold learning
methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time.time()
    trans_data = sklearn.manifold.LocallyLinearEmbedding(n_neighbors, 2, method=method).fit_transform(sphere_data).T
    t1 = time.time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    matplotlib.pyplot.scatter(trans_data[0], trans_data[1], c=colors, cmap=matplotlib.pyplot.cm.rainbow)
    matplotlib.pyplot.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    matplotlib.pyplot.axis('tight')

# Perform Isomap Manifold learning.
t0 = time.time()
trans_data = sklearn.manifold.Isomap(n_neighbors, n_components=2).fit_transform(sphere_data).T
t1 = time.time()
print("%s: %.2g sec" % ('ISO', t1 - t0))

ax = fig.add_subplot(257)
matplotlib.pyplot.scatter(trans_data[0], trans_data[1], c=colors, cmap=matplotlib.pyplot.cm.rainbow)
matplotlib.pyplot.title("%s (%.2g sec)" % ('Isomap', t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

# Perform Multi-dimensional scaling.
t0 = time.time()
mds = sklearn.manifold.MDS(2, max_iter=100, n_init=1)
trans_data = mds.fit_transform(sphere_data).T
t1 = time.time()
print("MDS: %.2g sec" % (t1 - t0))

ax = fig.add_subplot(258)
matplotlib.pyplot.scatter(trans_data[0], trans_data[1], c=colors, cmap=matplotlib.pyplot.cm.rainbow)
matplotlib.pyplot.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

# Perform Spectral Embedding.
t0 = time.time()
se = sklearn.manifold.SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
trans_data = se.fit_transform(sphere_data).T
t1 = time.time()
print("Spectral Embedding: %.2g sec" % (t1 - t0))

ax = fig.add_subplot(259)
matplotlib.pyplot.scatter(trans_data[0], trans_data[1], c=colors, cmap=matplotlib.pyplot.cm.rainbow)
matplotlib.pyplot.title("Spectral Embedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

# Perform t-distributed stochastic neighbor embedding.
t0 = time.time()
tsne = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
trans_data = tsne.fit_transform(sphere_data).T
t1 = time.time()
print("t-SNE: %.2g sec" % (t1 - t0))

ax = fig.add_subplot(2, 5, 10)
matplotlib.pyplot.scatter(trans_data[0], trans_data[1], c=colors, cmap=matplotlib.pyplot.cm.rainbow)
matplotlib.pyplot.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
matplotlib.pyplot.axis('tight')

matplotlib.pyplot.show()