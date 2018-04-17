import numpy
import matplotlib.pyplot
import sklearn.datasets
import sklearn.neighbors
import sklearn.decomposition
import sklearn.model_selection

# load the data
digits = sklearn.datasets.load_digits()
data = digits.data

# project the 64-dimensional data to a lower dimension
pca = sklearn.decomposition.PCA(n_components=15, whiten=False)
data = pca.fit_transform(digits.data)

# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': numpy.logspace(-1, 1, 20)}
grid = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

# sample 44 new points from the data
new_data = kde.sample(44, random_state=0)
new_data = pca.inverse_transform(new_data)

# turn data into a 4x11 grid
new_data = new_data.reshape((4, 11, -1))
real_data = digits.data[:44].reshape((4, 11, -1))

# plot real digits and resampled digits
fig, ax = matplotlib.pyplot.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    ax[4, j].set_visible(False)
    for i in range(4):
        im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)), cmap=matplotlib.pyplot.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)
        im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)), cmap=matplotlib.pyplot.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)

ax[0, 5].set_title('Selection from the input data')
ax[5, 5].set_title('"New" digits drawn from the kernel density model')

matplotlib.pyplot.show()