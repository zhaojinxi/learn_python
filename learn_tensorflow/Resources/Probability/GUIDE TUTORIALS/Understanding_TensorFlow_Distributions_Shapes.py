import collections
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Scalar Distributions
def describe_distributions(distributions):
    print('\n'.join([str(d) for d in distributions]))

poisson_distributions = [
    tfd.Poisson(rate=1., name='One Poisson Scalar Batch'),
    tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons'),
    tfd.Poisson(rate=[[1., 10., 100., ], [2., 20., 200.]],
                name='Two-by-Three Poissons'),
    tfd.Poisson(rate=[1.], name='One Poisson Vector Batch'),
    tfd.Poisson(rate=[[1.]], name='One Poisson Expanded Batch')
]

describe_distributions(poisson_distributions)

normal_distributions = [
    tfd.Normal(loc=0., scale=1., name='Standard'),
    tfd.Normal(loc=[0.], scale=1., name='Standard Vector Batch'),
    tfd.Normal(loc=[0., 1., 2., 3.], scale=1., name='Different Locs'),
    tfd.Normal(loc=[0., 1., 2., 3.], scale=[[1.], [5.]],
               name='Broadcasting Scale')
]

describe_distributions(normal_distributions)

describe_distributions(
    [tfd.Normal(loc=[[0., 1., 2., 3], [0., 1., 2., 3.]],
                scale=[[1., 1., 1., 1.], [5., 5., 5., 5.]])])

# Sampling Scalar Distributions
def describe_sample_tensor_shape(sample_shape, distribution):
    print('Sample shape:', sample_shape)
    print('Returned sample tensor shape:',
          distribution.sample(sample_shape).shape)

def describe_sample_tensor_shapes(distributions, sample_shapes):
    started = False
    for distribution in distributions:
      print(distribution)
      for sample_shape in sample_shapes:
        describe_sample_tensor_shape(sample_shape, distribution)
      print()

sample_shapes = [1, 2, [1, 5], [3, 4, 5]]
describe_sample_tensor_shapes(poisson_distributions, sample_shapes)

describe_sample_tensor_shapes(normal_distributions, sample_shapes)

# Computing log_prob For Scalar Distributions
three_poissons = tfd.Poisson(rate=[1., 10., 100.], name='Three Poissons')

three_poissons.log_prob([[1., 10., 100.], [100., 10., 1]])

three_poissons.log_prob([[[[1., 10., 100.], [100., 10., 1.]]]])

three_poissons.log_prob([10.])

three_poissons.log_prob([[[1.], [10.]], [[100.], [1000.]]])

poisson_2_by_3 = tfd.Poisson(
    rate=[[1., 10., 100., ], [2., 20., 200.]],
    name='Two-by-Three Poissons')

poisson_2_by_3.log_prob([1.])

poisson_2_by_3.log_prob([[1., 1., 1.], [1., 1., 1.]])

poisson_2_by_3.log_prob([[1., 10., 100.]])

poisson_2_by_3.log_prob([[1., 10., 100.], [1., 10., 100.]])

poisson_2_by_3.log_prob([[1., 1., 1.], [2., 2., 2.]])

poisson_2_by_3.log_prob([[1.], [2.]])

poisson_2_by_3.log_prob([[[1., 1., 1.], [1., 1., 1.]],
                         [[2., 2., 2.], [2., 2., 2.]]])

poisson_2_by_3.log_prob([[[1.], [1.]], [[2.], [2.]]])

poisson_2_by_3.log_prob([[[1., 1., 1.]], [[2., 2., 2.]]])

poisson_2_by_3.log_prob([[[1.]], [[2.]]])

poisson_2_by_3.log_prob(tf.constant([1., 2.])[..., tf.newaxis, tf.newaxis])

three_poissons.log_prob([[1.], [10.], [50.], [100.]])

three_poissons.log_prob(tf.constant([1., 10., 50., 100.])[..., tf.newaxis])

# Multivariate distributions
multinomial_distributions = [
    tfd.Multinomial(total_count=100., probs=[.5, .4, .1], name='One Multinomial'),
    tfd.Multinomial(total_count=[100., 1000.], probs=[.5, .4, .1], name='Two Multinomials Same Probs'),
    tfd.Multinomial(total_count=100., probs=[[.5, .4, .1], [.1, .2, .7]], name='Two Multinomials Same Counts'),
    tfd.Multinomial(total_count=[100., 1000.], probs=[[.5, .4, .1], [.1, .2, .7]], name='Two Multinomials Different Everything')]

describe_distributions(multinomial_distributions)
