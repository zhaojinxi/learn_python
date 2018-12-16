import matplotlib.pyplot as plt
import collections
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Basic Univariate Distributions
n = tfd.Normal(loc=0., scale=1.)

n.sample()

n.sample(3)

n.log_prob(0.)

n.log_prob([0., 2., 4.])

b = tfd.Bernoulli(probs=0.7)

b.sample()

b.sample(8)

b.log_prob(1)

b.log_prob([1, 0, 1, 0])

# Multivariate Distributions
nd = tfd.MultivariateNormalDiag(loc=[0., 10.], scale_diag=[1., 4.])

tfd.Normal(loc=0., scale=1.)

nd.sample()

nd.sample(5)

nd.log_prob([0., 10])

nd = tfd.MultivariateNormalFullCovariance(
    loc=[0., 5], covariance_matrix=[[1., .7], [.7, 1.]])
data = nd.sample(200)
plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.4)
plt.axis([-5, 5, 0, 10])
plt.title("Data set")
plt.show()

# Multiple Distributions
b3 = tfd.Bernoulli(probs=[.3, .5, .7])

b3.sample()

b3.sample(6)

b3.prob([1, 1, 0])

# Using Independent To Aggregate Batches to Events
b3_joint = tfd.Independent(b3, reinterpreted_batch_ndims=1)

b3_joint.prob([1, 1, 0])

tf.reduce_prod(b3.prob([1, 1, 0]))

# Batches of Multivariate Distirbutions
nd_batch = tfd.MultivariateNormalFullCovariance(
    loc=[[0., 0.], [1., 1.], [2., 2.]],
    covariance_matrix=[[[1., .1], [.1, 1.]],
                       [[1., .3], [.3, 1.]],
                       [[1., .5], [.5, 1.]]])

nd_batch.sample(4)

nd_batch.log_prob([[0., 0.], [1., 1.], [2., 2.]])

#Broadcasting, aka Why Is This So Confusing?
n = tfd.Normal(loc=0., scale=1.)

n.log_prob(0.)

n.log_prob([0.])

n.log_prob([[0., 1.], [-1., 2.]])

nd = tfd.MultivariateNormalDiag(loc=[0., 1.], scale_diag=[1., 1.])

nd.log_prob([0., 0.])

nd.log_prob([[0., 0.],
             [1., 1.],
             [2., 2.]])

nd.log_prob([0.])

nd.log_prob([[0.], [1.], [2.]])

b3 = tfd.Bernoulli(probs=[.3, .5, .7])

b3.prob([1])

b3.prob([[0], [1]])
