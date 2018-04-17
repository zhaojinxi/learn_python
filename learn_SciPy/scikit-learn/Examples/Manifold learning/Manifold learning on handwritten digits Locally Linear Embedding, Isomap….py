import time
import numpy
import matplotlib.pyplot
import matplotlib.offsetbox
import sklearn.manifold
import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.discriminant_analysis
import sklearn.random_projection

digits = sklearn.datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = numpy.min(X, 0), numpy.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(111)
    for i in range(X.shape[0]):
        matplotlib.pyplot.text(X[i, 0], X[i, 1], str(digits.target[i]), color=matplotlib.pyplot.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    if hasattr(matplotlib.offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = numpy.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = numpy.sum((X[i] - shown_images) ** 2, 1)
            if numpy.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = numpy.r_[shown_images, [X[i]]]
            imagebox = matplotlib.offsetbox.AnnotationBbox(matplotlib.offsetbox.OffsetImage(digits.images[i], cmap=matplotlib.pyplot.cm.gray_r), X[i])
            ax.add_artist(imagebox)
    matplotlib.pyplot.xticks([]), matplotlib.pyplot.yticks([])
    if title is not None:
        matplotlib.pyplot.title(title)

#----------------------------------------------------------------------
# Plot images of the digits
n_img_per_row = 20
img = numpy.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

matplotlib.pyplot.imshow(img, cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.xticks([])
matplotlib.pyplot.yticks([])
matplotlib.pyplot.title('A selection from the 64-dimensional digits dataset')

#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = sklearn.random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits")

#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time.time()
X_pca = sklearn.decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca, "Principal Components projection of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components

print("Computing Linear Discriminant Analysis projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time.time()
X_lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
plot_embedding(X_lda, "Linear Discriminant projection of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time.time()
X_iso = sklearn.manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding(X_iso, "Isomap projection of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = sklearn.manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
t0 = time.time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle, "Locally Linear Embedding of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = sklearn.manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
t0 = time.time()
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_mlle, "Modified Locally Linear Embedding of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
clf = sklearn.manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
t0 = time.time()
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_hlle, "Hessian Locally Linear Embedding of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = sklearn.manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
t0 = time.time()
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_ltsa, "Local Tangent Space Alignment of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = sklearn.manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time.time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds, "MDS embedding of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
print("Computing Totally Random Trees embedding")
hasher = sklearn.ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
t0 = time.time()
X_transformed = hasher.fit_transform(X)
pca = sklearn.decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding(X_reduced, "Random forest embedding of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# Spectral embedding of the digits dataset
print("Computing Spectral embedding")
embedder = sklearn.manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
t0 = time.time()
X_se = embedder.fit_transform(X)

plot_embedding(X_se, "Spectral embedding of the digits (time %.2fs)" % (time.time() - t0))

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time.time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time.time() - t0))

matplotlib.pyplot.show()