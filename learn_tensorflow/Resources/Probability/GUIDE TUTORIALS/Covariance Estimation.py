import collections
import math
import os
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# First attempt
# We're assuming 2-D data with a known true mean of (0, 0)
true_mean = np.zeros([2], dtype=np.float32)
# We'll make the 2 coordinates correlated
true_cor = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=np.float32)
# And we'll give the 2 coordinates different variances
true_var = np.array([4.0, 1.0], dtype=np.float32)
# Combine the variances and correlations into a covariance matrix
true_cov = np.expand_dims(np.sqrt(true_var), axis=1).dot(
    np.expand_dims(np.sqrt(true_var), axis=1).T) * true_cor
# We'll be working with precision matrices, so we'll go ahead and compute the true precision matrix here
true_precision = np.linalg.inv(true_cov)

# Here's our resulting covariance matrix
print(true_cov)
# Verify that it's positive definite, since np.random.multivariate_normal
# complains about it not being positive definite for some reason.
# (Note that I'll be including a lot of sanity checking code in this notebook - it's a *huge* help for debugging)
print('eigenvalues: ', np.linalg.eigvals(true_cov))

# Set the seed so the results are reproducible.
np.random.seed(123)
# Now generate some observations of our random variable.
# (Note that I'm suppressing a bunch of spurious about the covariance matrix
# not being positive semidefinite via check_valid='ignore' because it really is
# positive definite!)
my_data = np.random.multivariate_normal(
    mean=true_mean, cov=true_cov, size=100,
    check_valid='ignore').astype(np.float32)

# Do a scatter plot of the observations to make sure they look like what we
# expect (higher variance on the x-axis, y values strongly correlated with x)
plt.scatter(my_data[:, 0], my_data[:, 1], alpha=0.75)
plt.show()

print('mean of observations:', np.mean(my_data, axis=0))
print('true mean:', true_mean)

print('covariance of observations:\n', np.cov(my_data, rowvar=False))
print('true covariance:\n', true_cov)


def log_lik_data_numpy(precision, data):
    # np.linalg.inv is a really inefficient way to get the covariance matrix, but remember we don't care about speed here
    cov = np.linalg.inv(precision)
    rv = scipy.stats.multivariate_normal(true_mean, cov)
    return np.sum(rv.logpdf(data))
    # test case: compute the log likelihood of the data given the true parameters
log_lik_data_numpy(true_precision, my_data)

PRIOR_DF = 3
PRIOR_SCALE = np.eye(2, dtype=np.float32) / PRIOR_DF
def log_lik_prior_numpy(precision):
    rv = scipy.stats.wishart(df=PRIOR_DF, scale=PRIOR_SCALE)
    return rv.logpdf(precision)
# test case: compute the prior for the true parameters
log_lik_prior_numpy(true_precision)

n = my_data.shape[0]
nu_prior = PRIOR_DF
v_prior = PRIOR_SCALE
nu_posterior = nu_prior + n
v_posterior = np.linalg.inv(np.linalg.inv(v_prior) + my_data.T.dot(my_data))
posterior_mean = nu_posterior * v_posterior
v_post_diag = np.expand_dims(np.diag(v_posterior), axis=1)
posterior_sd = np.sqrt(nu_posterior *
                       (v_posterior ** 2.0 + v_post_diag.dot(v_post_diag.T)))

sample_precision = np.linalg.inv(np.cov(my_data, rowvar=False, bias=False))
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(10, 10)
for i in range(2):
  for j in range(2):
    ax = axes[i, j]
    loc = posterior_mean[i, j]
    scale = posterior_sd[i, j]
    xmin = loc - 3.0 * scale
    xmax = loc + 3.0 * scale
    x = np.linspace(xmin, xmax, 1000)
    y = scipy.stats.norm.pdf(x, loc=loc, scale=scale)
    ax.plot(x, y)
    ax.axvline(true_precision[i, j], color='red', label='True precision')
    ax.axvline(sample_precision[i, j], color='red',
               linestyle=':', label='Sample precision')
    ax.set_title('precision[%d, %d]' % (i, j))
plt.legend()
plt.show()

with tf.Graph().as_default() as g:
    # case 1: get log probabilities for a vector of iid draws from a single
    # normal distribution
    norm1 = tfd.Normal(loc=0., scale=1.)
    probs1 = norm1.log_prob(tf.constant([1., 0.5, 0.]))
    # case 2: get log probabilities for a vector of independent draws from
    # multiple normal distributions with different parameters.  Note the vector
    # values for loc and scale in the Normal constructor.
    norm2 = tfd.Normal(loc=[0., 2., 4.], scale=[1., 1., 1.])
    probs2 = norm2.log_prob(tf.constant([1., 0.5, 0.]))
    g.finalize()
with tf.Session(graph=g) as sess:
    print('iid draws from a single normal:', sess.run(probs1))
    print('draws from a batch of normals:', sess.run(probs2))

VALIDATE_ARGS = True
ALLOW_NAN_STATS = False


def log_lik_data(precisions, replicated_data):
    n = tf.shape(precisions)[0]  # number of precision matrices
  # We're estimating a precision matrix; we have to invert to get log probabilities.  Cholesky inversion should be relatively efficient, but as we'll see later, it's even better if we can avoid doing the Cholesky decomposition altogether.
    precisions_cholesky = tf.cholesky(precisions)
    covariances = tf.cholesky_solve(precisions_cholesky,
                                    tf.eye(2, batch_shape=[n]))
    rv_data = tfd.MultivariateNormalFullCovariance(
        loc=tf.zeros([n, 2]),
        covariance_matrix=covariances,
        validate_args=VALIDATE_ARGS,
        allow_nan_stats=ALLOW_NAN_STATS)

    return tf.reduce_sum(rv_data.log_prob(replicated_data), axis=0)


# For our test, we'll use a tensor of 2 precision matrices.
# We'll need to replicate our data for the likelihood function.
# Remember, TFP wants the data to be structured so that the sample dimensions are first (100 here), then the batch dimensions (2 here because we have 2 precision matrices), then the event dimensions (2 because we have 2-D Gaussian data).  We'll need to add a middle dimension for the batch using expand_dims, and then we'll need to create 2 replicates in this new dimension using tile.
n = 2
replicated_data = np.tile(np.expand_dims(my_data, axis=1), reps=[1, 2, 1])
print(replicated_data.shape)

# check against the numpy implementation
with tf.Graph().as_default() as g:
    precisions = np.stack([np.eye(2, dtype=np.float32), true_precision])
    n = precisions.shape[0]
    lik_tf = log_lik_data(precisions, replicated_data=replicated_data)
    g.finalize()
with tf.Session(graph=g) as sess:
    lik_tf_ = sess.run(lik_tf)
    for i in range(n):
        print(i)
        print('numpy:', log_lik_data_numpy(precisions[i], my_data))
        print('tensorflow:', lik_tf_[i])


def log_lik_prior(precisions):
    rv_precision = tfd.Wishart(
        df=PRIOR_DF,
        scale=PRIOR_SCALE,
        validate_args=VALIDATE_ARGS,
        allow_nan_stats=ALLOW_NAN_STATS)
    return rv_precision.log_prob(precisions)


# check against the numpy implementation
with tf.Graph().as_default() as g:
    precisions = np.stack([np.eye(2, dtype=np.float32), true_precision])
    n = precisions.shape[0]
    lik_tf = log_lik_prior(precisions)
    g.finalize()
with tf.Session(graph=g) as sess:
    lik_tf_ = sess.run(lik_tf)
    for i in range(n):
        print(i)
        print('numpy:', log_lik_prior_numpy(precisions[i]))
        print('tensorflow:', lik_tf_[i])


def get_log_lik(data, n_chains=1):
    # The data argument that is passed in will be available to the inner function
    # below so it doesn't have to be passed in as a parameter.
    replicated_data = np.tile(np.expand_dims(
        data, axis=1), reps=[1, n_chains, 1])

    def _log_lik(precision):
        return log_lik_data(precision, replicated_data) + log_lik_prior(precision)

    return _log_lik


with tf.Graph().as_default() as g:
    # Use expand_dims because we want to pass in a tensor of starting values
    init_precision = tf.expand_dims(tf.eye(2), axis=0)
    log_lik_fn = get_log_lik(my_data, n_chains=1)

    # we'll just do a few steps here
    num_results = 10
    num_burnin_steps = 10

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
            init_precision,
        ],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_lik_fn,
            step_size=0.1,
            num_leapfrog_steps=3,
            seed=123),
        parallel_iterations=1)

    g.finalize()
with tf.Session(graph=g) as sess:
    tf.set_random_seed(123)

    try:
        states_, kernel_results_ = sess.run([states, kernel_results])
    except Exception as e:
        # shorten the giant stack trace
        lines = str(e).split('\n')
        print('\n'.join(lines[:5]+['...']+lines[-3:]))


def get_log_lik_verbose(data, n_chains=1):
    # The data argument that is passed in will be available to the inner function
    # below so it doesn't have to be passed in as a parameter.
    replicated_data = np.tile(np.expand_dims(
        data, axis=1), reps=[1, n_chains, 1])

    def _log_lik(precisions):
        # An internal method we'll make into a TensorFlow operation via tf.py_func
        def _print_precisions(precisions):
            print('precisions:\n', precisions)
            return False  # operations must return something!
        # Turn our method into a TensorFlow operation
        print_op = tf.py_func(_print_precisions, [precisions], tf.bool)

        # Assertions are also operations, and some care needs to be taken to ensure
        # that they're executed
        assert_op = tf.assert_equal(
            precisions, tf.transpose(precisions, perm=[0, 2, 1]), data=[precisions],
            message='not symmetrical', summarize=4, name='symmetry_check')

        # The control_dependencies statement forces its arguments to be executed
        # before subsequent operations
        with tf.control_dependencies([print_op, assert_op]):
            return (log_lik_data(precisions, replicated_data) +
                log_lik_prior(precisions))

    return _log_lik


with tf.Graph().as_default() as g:
    tf.set_random_seed(123)
    init_precision = tf.expand_dims(tf.eye(2), axis=0)
    log_lik_fn = get_log_lik_verbose(my_data)

    # we'll just do a few steps here
    num_results = 10
    num_burnin_steps = 10

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[
            init_precision,
        ],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_lik_fn,
            step_size=0.1,
            num_leapfrog_steps=3,
            seed=123),
        parallel_iterations=1)

    g.finalize()
with tf.Session(graph=g) as sess:
    try:
        states_, kernel_results_ = sess.run([states, kernel_results])
    except Exception as e:
        # shorten the giant stack trace
        lines = str(e).split('\n')
        print('\n'.join(lines[:5]+['...']+lines[-3:]))

# Version 2: reparametrizing to unconstrained parameters
# Our transform has 3 stages that we chain together via composition:
precision_to_unconstrained = tfb.Chain([
    # step 3: flatten the lower triangular portion of the matrix
    tfb.Invert(tfb.FillTriangular(validate_args=VALIDATE_ARGS)),
    # step 2: take the log of the diagonals
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp(validate_args=VALIDATE_ARGS))),
    # step 1: decompose the precision matrix into its Cholesky factors
    tfb.Invert(tfb.CholeskyOuterProduct(validate_args=VALIDATE_ARGS)),
])

# sanity checks
with tf.Graph().as_default() as g:
  m = tf.constant([[1., 2.], [2., 8.]])
  m_fwd = precision_to_unconstrained.forward(m)
  m_inv = precision_to_unconstrained.inverse(m_fwd)

  # bijectors handle tensors of values, too!
  m2 = tf.stack([m, tf.eye(2)])
  m2_fwd = precision_to_unconstrained.forward(m2)
  m2_inv = precision_to_unconstrained.inverse(m2_fwd)

  g.finalize()

with tf.Session(graph=g) as sess:
    m_, m_fwd_, m_inv_ = sess.run([m, m_fwd, m_inv])
    print('single input:')
    print('m:\n', m_)
    print('precision_to_unconstrained(m):\n', m_fwd_)
    print('inverse(precision_to_unconstrained(m)):\n', m_inv_)

    m2_, m2_fwd_, m2_inv_ = sess.run([m2, m2_fwd, m2_inv])
    print('tensor of inputs:')
    print('m2:\n', m2_)
    print('precision_to_unconstrained(m2):\n', m2_fwd_)
    print('inverse(precision_to_unconstrained(m2)):\n',m2_inv_)


def log_lik_prior_transformed(transformed_precisions):
  rv_precision = tfd.TransformedDistribution(
      tfd.Wishart(
          df=PRIOR_DF,
          scale=PRIOR_SCALE,
          validate_args=VALIDATE_ARGS,
          allow_nan_stats=ALLOW_NAN_STATS),
      bijector=precision_to_unconstrained,
      validate_args=VALIDATE_ARGS)
  return rv_precision.log_prob(transformed_precisions)


# Check against the numpy implementation.  Note that when comparing, we need
# to add in the Jacobian correction.
with tf.Graph().as_default() as g:
  precisions = np.stack([np.eye(2, dtype=np.float32), true_precision])
  transformed_precisions = precision_to_unconstrained.forward(precisions)
  lik_tf = log_lik_prior_transformed(transformed_precisions)
  corrections = precision_to_unconstrained.inverse_log_det_jacobian(
      transformed_precisions, event_ndims=1)
  n = precisions.shape[0]
  g.finalize()

with tf.Session(graph=g) as sess:
  lik_tf_, corrections_ = sess.run([lik_tf, corrections])
  for i in range(n):
    print(i)
    print('numpy:', log_lik_prior_numpy(precisions[i]) + corrections_[i])
    print('tensorflow:', lik_tf_[i])


def log_lik_data_transformed(transformed_precisions, replicated_data):
  # We recover the precision matrix by inverting our bijector.  This is
  # inefficient since we really want the Cholesky decomposition of the
  # precision matrix, and the bijector has that in hand during the inversion,
  # but we'll worry about efficiency later.
  n = tf.shape(transformed_precisions)[0]
  precisions = precision_to_unconstrained.inverse(transformed_precisions)
  precisions_cholesky = tf.cholesky(precisions)
  covariances = tf.cholesky_solve(precisions_cholesky,
                                  tf.eye(2, batch_shape=[n]))
  rv_data = tfd.MultivariateNormalFullCovariance(
      loc=tf.zeros([n, 2]),
      covariance_matrix=covariances,
      validate_args=VALIDATE_ARGS,
      allow_nan_stats=ALLOW_NAN_STATS)

  return tf.reduce_sum(rv_data.log_prob(replicated_data), axis=0)


# sanity check
with tf.Graph().as_default() as g:
  precisions = np.stack([np.eye(2, dtype=np.float32), true_precision])
  transformed_precisions = precision_to_unconstrained.forward(precisions)
  lik_tf = log_lik_data_transformed(transformed_precisions, replicated_data)
  g.finalize()

with tf.Session(graph=g) as sess:
  lik_tf_ = sess.run(lik_tf)
  for i in range(precisions.shape[0]):
    print(i)
    print('numpy:', log_lik_data_numpy(precisions[i], my_data))
    print('tensorflow:', lik_tf_[i])


def get_log_lik_transformed(data, n_chains=1):
  # The data argument that is passed in will be available to the inner function
  # below so it doesn't have to be passed in as a parameter.
  replicated_data = np.tile(np.expand_dims(
      data, axis=1), reps=[1, n_chains, 1])

  def _log_lik_transformed(transformed_precisions):
    return (log_lik_data_transformed(transformed_precisions, replicated_data) +
            log_lik_prior_transformed(transformed_precisions))

  return _log_lik_transformed


# make sure everything runs
with tf.Graph().as_default() as g:
  log_lik_fn = get_log_lik_transformed(my_data)
  m = tf.expand_dims(tf.eye(2), axis=0)
  lik = log_lik_fn(precision_to_unconstrained.forward(m))
  g.finalize()

with tf.Session(graph=g) as sess:
    print(sess.run(lik))

# We'll choose a proper random initial value this time
np.random.seed(123)
initial_value_cholesky = np.array([[0.5 + np.random.uniform(), 0.0], [-0.5 + np.random.uniform(), 0.5 + np.random.uniform()]], dtype=np.float32)
initial_value = np.expand_dims(
    initial_value_cholesky.dot(initial_value_cholesky.T), axis=0)

# The sampler works with unconstrained values, so we'll transform our initial
# value
with tf.Graph().as_default() as g:
  initial_value_transformed = precision_to_unconstrained.forward(initial_value)
  g.finalize()

with tf.Session(graph=g) as sess:
  initial_value_transformed_ = sess.run(initial_value_transformed)

# Sample!
with tf.Graph().as_default() as g:
  tf.set_random_seed(123)
  log_lik_fn = get_log_lik_transformed(my_data, n_chains=1)

  num_results = 1000
  num_burnin_steps = 1000

  states, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=[
          initial_value_transformed_,
      ],
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=log_lik_fn,
          step_size=0.1,
          num_leapfrog_steps=3,
          seed=123),
      parallel_iterations=1)
  # transform samples back to their constrained form
  precision_samples = precision_to_unconstrained.inverse(states)

  g.finalize()

with tf.Session(graph=g) as sess:
  states_, precision_samples_, kernel_results_ = sess.run(
      [states, precision_samples, kernel_results])

print('True posterior mean:\n', posterior_mean)
print('Sample mean:\n', np.mean(np.reshape(precision_samples_, [-1, 2, 2]), axis=0))

np.reshape(precision_samples_, [-1, 2, 2])

# Look at the acceptance for the last 100 samples
print(np.squeeze(kernel_results_.is_accepted)[-100:])
print('Fraction of samples accepted:', np.mean(np.squeeze(kernel_results_.is_accepted)))

# Version 3: sampling with an adaptive step size
# The number of chains is determined by the shape of the initial values.
# Here we'll generate 3 chains, so we'll need a tensor of 3 initial values.
N_CHAINS = 3

np.random.seed(123)

initial_values = []
for i in range(N_CHAINS):
  initial_value_cholesky = np.array(
      [[0.5 + np.random.uniform(), 0.0],
       [-0.5 + np.random.uniform(), 0.5 + np.random.uniform()]],
      dtype=np.float32)
  initial_values.append(initial_value_cholesky.dot(initial_value_cholesky.T))
initial_values = np.stack(initial_values)

# Transform our initial values to their unconstrained form
# (Transforming the value in its own session is a workaround for b/72831017)
with tf.Graph().as_default() as g:
  initial_values_transformed = precision_to_unconstrained.forward(
      initial_values)
  g.finalize()

with tf.Session(graph=g) as sess:
  initial_values_transformed_ = sess.run(initial_values_transformed)

# Code adapted from tensorflow_probability/python/mcmc/hmc.py
with tf.Graph().as_default() as g:
  tf.set_random_seed(123)
  log_lik_fn = get_log_lik_transformed(my_data)

  # Tuning acceptance rates:
  dtype = np.float32
  num_warmup_iter = 2500
  num_chain_iter = 2500

  # Set the target average acceptance ratio for the HMC as suggested by
  # Beskos et al. (2013):
  # https://projecteuclid.org/download/pdfview_1/euclid.bj/1383661192
  target_accept_rate = 0.651

  x = tf.get_variable(name='x', initializer=initial_values_transformed_)
  step_size = tf.get_variable(name='step_size',
                              initializer=tf.constant(0.01, dtype=dtype))

  # Initialize the HMC sampler.
  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=log_lik_fn,
      step_size=step_size,
      num_leapfrog_steps=3)

  # One iteration of the HMC
  next_x, other_results = hmc.one_step(
      current_state=x,
      previous_kernel_results=hmc.bootstrap_results(x))

  x_update = x.assign(next_x)
  precision = precision_to_unconstrained.inverse(x_update)

  # Adapt the step size using standard adaptive MCMC procedure. See Section 4.2
  # of Andrieu and Thoms (2008):
  # http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf

  # NOTE: One important change we need to make from the hmc.py version is to
  # combine the log_accept_ratio values from the different chains when
  # deciding how to update the step size.  Here we use the mean
  # log_accept_ratio to decide.
  step_size_update = step_size.assign_add(
      step_size * tf.where(
          tf.exp(tf.minimum(tf.reduce_mean(
              other_results.log_accept_ratio), 0.)) >
          target_accept_rate,
          x=0.1, y=-0.1))

  # Note, the adaptations are performed during warmup only.
  warmup = tf.group([x_update, step_size_update])

  init = tf.global_variables_initializer()

  g.finalize()

with tf.Session(graph=g) as sess:
  # Initialize variables
  sess.run(init)

  # Warm up the sampler and adapt the step size
  print('Warmup')
  start_time = time.time()
  for i in range(num_warmup_iter):
    sess.run(warmup)
    if i % 500 == 0:
      print('Step %d' % i)
  end_time = time.time()
  print('Time per step:', (end_time - start_time) / num_warmup_iter)
  print('Step size: %g' % sess.run(step_size))

  # Collect samples without adapting step size
  print('Sampling')
  start_time = time.time()
  packed_samples = np.zeros([num_chain_iter, N_CHAINS, 3])
  precision_samples = np.zeros([num_chain_iter, N_CHAINS, 2, 2])
  results = []
  for i in range(num_chain_iter):
    _, x_, precision_, other_results_ = sess.run(
        [x_update, x, precision, other_results])
    packed_samples[i, :] = x_
    precision_samples[i, :] = precision_
    results.append(other_results_)
    if i % 500 == 0:
      print('Step %d' % i)
  end_time = time.time()
  print('Time per step:', (end_time - start_time) / num_chain_iter)

is_accepted = np.array([r.is_accepted for r in results])
print(np.mean(is_accepted))

precision_samples_reshaped = np.reshape(precision_samples, [-1, 2, 2])

print('True posterior mean:\n', posterior_mean)
print('Mean of samples:\n', np.mean(precision_samples_reshaped, axis=0))

print('True posterior standard deviation:\n', posterior_sd)
print('Standard deviation of samples:\n', np.std(precision_samples_reshaped, axis=0))

with tf.Graph().as_default() as g:
  r_hat = tfp.mcmc.potential_scale_reduction(precision_samples)
  g.finalize()

with tf.Session(graph=g) as sess:
  print(sess.run(r_hat))

fig, axes = plt.subplots(2, 2, sharey=True)
fig.set_size_inches(8, 8)
for i in range(2):
  for j in range(2):
    ax = axes[i, j]
    ax.hist(precision_samples_reshaped[:, i, j])
    ax.axvline(true_precision[i, j], color='red',
               label='True precision')
    ax.axvline(sample_precision[i, j], color='red', linestyle=':',
               label='Sample precision')
    ax.set_title('precision[%d, %d]' % (i, j))
plt.tight_layout()
plt.legend()
plt.show()

fig, axes = plt.subplots(4, 4)
fig.set_size_inches(12, 12)
for i1 in range(2):
  for j1 in range(2):
    index1 = 2 * i1 + j1
    for i2 in range(2):
      for j2 in range(2):
        index2 = 2 * i2 + j2
        ax = axes[index1, index2]
        ax.scatter(precision_samples_reshaped[:, i1, j1],
                   precision_samples_reshaped[:, i2, j2], alpha=0.1)
        ax.axvline(true_precision[i1, j1], color='red')
        ax.axhline(true_precision[i2, j2], color='red')
        ax.axvline(sample_precision[i1, j1], color='red', linestyle=':')
        ax.axhline(sample_precision[i2, j2], color='red', linestyle=':')
        ax.set_title('(%d, %d) vs (%d, %d)' % (i1, j1, i2, j2))
plt.tight_layout()
plt.show()

# Version 4: simpler sampling of constrained parameters
# The bijector we need for the TransformedTransitionKernel is the inverse of
# the one we used above
unconstrained_to_precision = tfb.Chain([
    # step 3: take the product of Cholesky factors
    tfb.CholeskyOuterProduct(validate_args=VALIDATE_ARGS),
    # step 2: exponentiate the diagonals
    tfb.TransformDiagonal(tfb.Exp(validate_args=VALIDATE_ARGS)),
    # step 3: map a vector to a lower triangular matrix
    tfb.FillTriangular(validate_args=VALIDATE_ARGS),
])

# quick sanity check
with tf.Graph().as_default() as g:
  m = tf.constant([[1., 2.], [2., 8.]])
  m_inv = unconstrained_to_precision.inverse(m)
  m_fwd = unconstrained_to_precision.forward(m_inv)
  g.finalize()

with tf.Session(graph=g) as sess:
  m_, m_inv_, m_fwd_ = sess.run([m, m_inv, m_fwd])
  print('m:\n', m_)
  print('unconstrained_to_precision.inverse(m):\n', m_inv_)
  print('forward(unconstrained_to_precision.inverse(m)):\n', m_fwd_)

# Code adapted from tensorflow_probability/python/mcmc/hmc.py
with tf.Graph().as_default() as g:
  tf.set_random_seed(123)
  log_lik_fn = get_log_lik(my_data)

  # Tuning acceptance rates:
  dtype = np.float32
  num_warmup_iter = 2500
  num_chain_iter = 2500

  # Set the target average acceptance ratio for the HMC as suggested by
  # Beskos et al. (2013):
  # https://projecteuclid.org/download/pdfview_1/euclid.bj/1383661192
  target_accept_rate = 0.651

  x = tf.get_variable(name='x', initializer=initial_values)
  step_size = tf.get_variable(
      name='step_size', initializer=tf.constant(0.01, dtype=dtype))

  # Initialize the HMC sampler, now wrapped in the TransformedTransitionKernel
  ttk = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=log_lik_fn,
          step_size=step_size,
          num_leapfrog_steps=3),
      bijector=[unconstrained_to_precision])

  # One iteration
  next_x, other_results = ttk.one_step(
      current_state=x,
      previous_kernel_results=ttk.bootstrap_results(x))

  x_update = x.assign(next_x)

  # Adapt the step size using standard adaptive MCMC procedure. See Section 4.2
  # of Andrieu and Thoms (2008):
  # http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf

  # NOTE: one change from above is that we have to look at
  # other_results.inner_results.log_accept_ratio, since the new kernel
  # wraps the results from the HMC kernel.
  step_size_update = step_size.assign_add(
      step_size * tf.where(
          tf.exp(tf.minimum(tf.reduce_mean(
              other_results.inner_results.log_accept_ratio), 0.)) >
          target_accept_rate,
          x=0.1, y=-0.1))

  # Note, the adaptations are performed during warmup only.
  warmup = tf.group([x_update, step_size_update])

  init = tf.global_variables_initializer()

  g.finalize()

with tf.Session(graph=g) as sess:
  # Initialize
  sess.run(init)

  # Warm up the sampler and adapt the step size
  print('Warmup')
  start_time = time.time()
  for i in range(num_warmup_iter):
    sess.run(warmup)
    if i % 500 == 0:
      print('Step %d' % i)
  end_time = time.time()
  print('Time per step:', (end_time - start_time) / num_warmup_iter)
  print('Step size: %g' % sess.run(step_size))

  # Collect samples without adapting step size
  print('Sampling')
  start_time = time.time()
  precision_samples = np.zeros([num_chain_iter, N_CHAINS, 2, 2])
  results = []
  for i in range(num_chain_iter):
    _, x_, other_results_ = sess.run([x_update, x, other_results])
    precision_samples[i, :] = x_
    results.append(other_results_)
    if i % 500 == 0:
      print('Step %d' % i)
  end_time = time.time()
  print('Time per step:', (end_time - start_time) / num_chain_iter)

with tf.Graph().as_default() as g:
  r_hat = tfp.mcmc.potential_scale_reduction(precision_samples)
  g.finalize()

with tf.Session(graph=g) as sess:
  print(sess.run(r_hat))

# The output samples have shape [n_steps, n_chains, 2, 2]
# Flatten them to [n_steps * n_chains, 2, 2] via reshape:
precision_samples_reshaped = np.reshape(precision_samples, [-1, 2, 2])

print('True posterior mean:\n', posterior_mean)
print('Mean of samples:\n', np.mean(precision_samples_reshaped, axis=0))

print('True posterior standard deviation:\n', posterior_sd)
print('Standard deviation of samples:\n', np.std(precision_samples_reshaped, axis=0))

# Optimizations
# An optimized Wishart distribution that has been transformed to operate on
# Cholesky factors instead of full matrices.  Note that we gain a modest
# additional speedup by specifying the Cholesky factor of the scale matrix
# (i.e. by passing in the scale_tril parameter instead of scale).


class CholeskyWishart(tfd.TransformedDistribution):
  """Wishart distribution reparameterized to use Cholesky factors."""

  def __init__(self,
               df,
               scale_tril,
               validate_args=False,
               allow_nan_stats=True,
               name='CholeskyWishart'):
    # Wishart has a bunch of methods that we want to support but not
    # implement.  We'll subclass TransformedDistribution here to take care of
    # those.  We'll override the few for which speed is critical and implement
    # them with a separate Wishart for which input_output_cholesky=True
    super(CholeskyWishart, self).__init__(
        distribution=tfd.Wishart(
            df=df,
            scale_tril=scale_tril,
            input_output_cholesky=False,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats),
        bijector=tfb.Invert(tfb.CholeskyOuterProduct()),
        validate_args=validate_args,
        name=name
    )
    # Here's the Cholesky distribution we'll use for log_prob() and sample()
    self.cholesky = tfd.Wishart(
        df=df,
        scale_tril=scale_tril,
        input_output_cholesky=True,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats)

  def _log_prob(self, x):
    return (self.cholesky.log_prob(x) +
            self.bijector.inverse_log_det_jacobian(x, event_ndims=2))

  def _sample_n(self, n, seed=None):
    return self.cholesky._sample_n(n, seed)


# some checks
PRIOR_SCALE_CHOLESKY = np.linalg.cholesky(PRIOR_SCALE)

with tf.Graph().as_default() as g:
  w_transformed = tfd.TransformedDistribution(
      tfd.Wishart(df=PRIOR_DF, scale_tril=PRIOR_SCALE_CHOLESKY),
      bijector=tfb.Invert(tfb.CholeskyOuterProduct()))
  w_optimized = CholeskyWishart(
      df=PRIOR_DF, scale_tril=PRIOR_SCALE_CHOLESKY)

  m = tf.placeholder(dtype=tf.float32)
  log_prob_transformed = w_transformed.log_prob(m)
  log_prob_optimized = w_optimized.log_prob(m)

  g.finalize()

with tf.Session(graph=g) as sess:
  for matrix in [np.eye(2, dtype=np.float32),
                 np.array([[1., 0.], [2., 8.]], dtype=np.float32)]:
    log_prob_transformed_, log_prob_optimized_ = sess.run(
        [log_prob_transformed, log_prob_optimized],
        feed_dict={m: matrix})
    print('Transformed Wishart:', log_prob_transformed_)
    print('Optimized Wishart', log_prob_optimized_)

# Here's our permuting bijector:
def get_permuter():
  permutation = [1, 0]
  return tfb.Chain([
      tfb.Transpose(rightmost_transposed_ndims=2),
      tfb.Permute(permutation=permutation),
      tfb.Transpose(rightmost_transposed_ndims=2),
      tfb.Permute(permutation=permutation),   
  ])


# Some sanity checks
with tf.Graph().as_default() as g:
  m = np.array([[1., 0.], [2., 8.]], dtype=np.float32)
  permuter = get_permuter()
  p_fwd = permuter.forward(m)
  p_fwd_fwd = permuter.forward(p_fwd)
  g.finalize()

with tf.Session(graph=g) as sess:
  print('m =\n', m)
  print('permuted = \n', sess.run(p_fwd))
  print('permuted^2 = \n', sess.run(p_fwd_fwd))

def get_wishart_cholesky_to_iw_cholesky():
  return tfb.Chain([
      # step 6: get the Cholesky factor for the covariance matrix
      tfb.Invert(tfb.CholeskyOuterProduct()),
      # step 5: undo our permutation (note that permuter.inverse = permuter.forward)
      get_permuter(),
      # step 4: form the covariance matrix from the inverted Cholesky factors
      tfb.CholeskyOuterProduct(),
      # step 3: make things lower triangular
      get_permuter(),
      # step 2: transpose the inverse
      tfb.Transpose(rightmost_transposed_ndims=2),
      # step 1: invert the Cholesky factor (see code below)
      tfb.MatrixInverseTriL()
  ])


# verify that the bijector works
with tf.Graph().as_default() as g:
  m = np.array([[1., 0.], [2., 8.]], dtype=np.float32)
  c_inv = m.dot(m.T)
  c = np.linalg.inv(c_inv)
  c_chol = np.linalg.cholesky(c)
  wishart_cholesky_to_iw_cholesky = get_wishart_cholesky_to_iw_cholesky()
  w_fwd = wishart_cholesky_to_iw_cholesky.forward(m)
  g.finalize()

with tf.Session(graph=g) as sess:
  print('numpy =\n', c_chol)
  print('bijector =\n', sess.run(w_fwd))

inverse_wishart_cholesky = tfd.TransformedDistribution(
    distribution=CholeskyWishart(
        df=PRIOR_DF,
        scale_tril=np.linalg.cholesky(np.linalg.inv(PRIOR_SCALE))),
    bijector=get_wishart_cholesky_to_iw_cholesky())

# Final(!) Version: using the Cholesky decomposition of the precision matrix
# Our new prior.
PRIOR_SCALE_CHOLESKY = np.linalg.cholesky(PRIOR_SCALE)


def log_lik_prior_cholesky(precisions_cholesky):
  rv_precision = CholeskyWishart(
      df=PRIOR_DF,
      scale_tril=PRIOR_SCALE_CHOLESKY,
      validate_args=VALIDATE_ARGS,
      allow_nan_stats=ALLOW_NAN_STATS)
  return rv_precision.log_prob(precisions_cholesky)


# Check against the slower TF implementation and the NumPy implementation.
# Note that when comparing to NumPy, we need to add in the Jacobian correction.
with tf.Graph().as_default() as g:
  precisions = [np.eye(2, dtype=np.float32),
                true_precision]
  precisions_cholesky = np.stack([np.linalg.cholesky(m) for m in precisions])
  precisions = np.stack(precisions)
  lik_tf = log_lik_prior_cholesky(precisions_cholesky)
  lik_tf_slow = tfd.TransformedDistribution(
      distribution=tfd.Wishart(df=PRIOR_DF, scale=PRIOR_SCALE),
      bijector=tfb.Invert(tfb.CholeskyOuterProduct())).log_prob(
      precisions_cholesky)
  corrections = tfb.Invert(tfb.CholeskyOuterProduct()).inverse_log_det_jacobian(
      precisions_cholesky, event_ndims=2)
  n = precisions.shape[0]
  g.finalize()

with tf.Session(graph=g) as sess:
  lik_tf_, lik_tf_slow_, corrections_ = sess.run(
      [lik_tf, lik_tf_slow, corrections])
  for i in range(n):
    print(i)
    print('numpy:', log_lik_prior_numpy(precisions[i]) + corrections_[i])
    print('tensorflow slow:', lik_tf_slow_[i])
    print('tensorflow fast:', lik_tf_[i])


class MVNPrecisionCholesky(tfd.TransformedDistribution):
  """Multivariate normal parametrized by loc and Cholesky precision matrix."""

  def __init__(self, loc, precision_cholesky, name=None):
    super(MVNPrecisionCholesky, self).__init__(
        distribution=tfd.Independent(
            tfd.Normal(loc=tf.zeros_like(loc),
                       scale=tf.ones_like(loc)),
            reinterpreted_batch_ndims=1),
        bijector=tfb.Chain([
            tfb.Affine(shift=loc),
            tfb.Invert(tfb.Affine(scale_tril=precision_cholesky,
                                  adjoint=True)),
        ]),
        name=name)


def log_lik_data_cholesky(precisions_cholesky, replicated_data):
  n = tf.shape(precisions_cholesky)[0]  # number of precision matrices
  rv_data = MVNPrecisionCholesky(
      loc=tf.zeros([n, 2]),
      precision_cholesky=precisions_cholesky)
  return tf.reduce_sum(rv_data.log_prob(replicated_data), axis=0)


# check against the numpy implementation
with tf.Graph().as_default() as g:
  true_precision_cholesky = np.linalg.cholesky(true_precision)
  precisions = [np.eye(2, dtype=np.float32), true_precision]
  precisions_cholesky = np.stack([np.linalg.cholesky(m) for m in precisions])
  precisions = np.stack(precisions)
  n = precisions_cholesky.shape[0]
  replicated_data = np.tile(np.expand_dims(my_data, axis=1), reps=[1, 2, 1])
  lik_tf = log_lik_data_cholesky(precisions_cholesky, replicated_data)
  g.finalize()

with tf.Session(graph=g) as sess:
  lik_tf_ = sess.run(lik_tf)
  for i in range(n):
    print(i)
    print('numpy:', log_lik_data_numpy(precisions[i], my_data))
    print('tensorflow:', lik_tf_[i])


def get_log_lik_cholesky(data, n_chains=1):
  # The data argument that is passed in will be available to the inner function
  # below so it doesn't have to be passed in as a parameter.
  replicated_data = np.tile(np.expand_dims(
      data, axis=1), reps=[1, n_chains, 1])

  def _log_lik_cholesky(precisions_cholesky):
    return (log_lik_data_cholesky(precisions_cholesky, replicated_data) +
            log_lik_prior_cholesky(precisions_cholesky))

  return _log_lik_cholesky


unconstrained_to_precision_cholesky = tfb.Chain([
    # step 2: exponentiate the diagonals
    tfb.TransformDiagonal(tfb.Exp(validate_args=VALIDATE_ARGS)),
    # step 1: expand the vector to a lower triangular matrix
    tfb.FillTriangular(validate_args=VALIDATE_ARGS),
])

# some checks
with tf.Graph().as_default() as g:
  inv = unconstrained_to_precision_cholesky.inverse(precisions_cholesky)
  fwd = unconstrained_to_precision_cholesky.forward(inv)
  g.finalize()

with tf.Session(graph=g) as sess:
  inv_, fwd_ = sess.run([inv, fwd])
  print('precisions_cholesky:\n', precisions_cholesky)
  print('\ninv:\n', inv_)
  print('\nfwd(inv):\n', fwd_)

# The number of chains is determined by the shape of the initial values.
# Here we'll generate 3 chains, so we'll need a tensor of 3 initial values.
N_CHAINS = 3

np.random.seed(123)

initial_values_cholesky = []
for i in range(N_CHAINS):
  initial_values_cholesky.append(np.array(
      [[0.5 + np.random.uniform(), 0.0],
       [-0.5 + np.random.uniform(), 0.5 + np.random.uniform()]],
      dtype=np.float32))
initial_values_cholesky = np.stack(initial_values_cholesky)

# Code adapted from tensorflow_probability/python/mcmc/hmc.py
with tf.Graph().as_default() as g:
  tf.set_random_seed(123)
  log_lik_fn = get_log_lik_cholesky(my_data)

  # Tuning acceptance rates:
  dtype = np.float32
  num_warmup_iter = 2500
  num_chain_iter = 2500

  # Set the target average acceptance ratio for the HMC as suggested by
  # Beskos et al. (2013):
  # https://projecteuclid.org/download/pdfview_1/euclid.bj/1383661192
  target_accept_rate = 0.651

  x = tf.get_variable(name='x', initializer=initial_values_cholesky)
  step_size = tf.get_variable(name='step_size',
                              initializer=tf.constant(0.01, dtype=dtype))

  # Initialize the HMC sampler, now wrapped in the TransformedTransitionKernel
  ttk = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=log_lik_fn,
      step_size=step_size,
      num_leapfrog_steps=3),
      bijector=[unconstrained_to_precision_cholesky])

  # One iteration
  next_x, other_results = ttk.one_step(
      current_state=x,
      previous_kernel_results=ttk.bootstrap_results(x))

  x_update = x.assign(next_x)
  precision = tf.matmul(x, x, transpose_b=True)

  # Adapt the step size using standard adaptive MCMC procedure. See Section 4.2
  # of Andrieu and Thoms (2008):
  # http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf

  # NOTE: one change from above is that we have to look at 
  # other_results.inner_results.log_accept_ratio, since the new kernel
  # wraps the results from the HMC kernel.
  step_size_update = step_size.assign_add(
      step_size * tf.where(
          tf.exp(tf.minimum(tf.reduce_mean(
              other_results.inner_results.log_accept_ratio), 0.)) >
              target_accept_rate,
          x=0.1, y=-0.1))

  # Note, the adaptations are performed during warmup only.
  warmup = tf.group([x_update, step_size_update])

  init = tf.global_variables_initializer()
  
  g.finalize()

with tf.Session(graph=g) as sess:
  # Initialize
  sess.run(init)

  # Warm up the sampler and adapt the step size
  print('Warmup')
  start_time = time.time()
  for i in range(num_warmup_iter):
    sess.run(warmup)
    if i % 500 == 0:
      print('Step %d' % i)
  end_time = time.time()
  print('Time per step:', (end_time - start_time) / num_warmup_iter)
  print('Step size: %g' % sess.run(step_size))

  # Collect samples without adapting step size
  print('Sampling')
  start_time = time.time()
  precision_samples = np.zeros([num_chain_iter, N_CHAINS, 2, 2])
  results = []
  for i in range(num_chain_iter):
    _, precision_, other_results_ = sess.run(
        [x_update, precision, other_results])
    precision_samples[i, :] = precision_
    results.append(other_results_)
    if i % 500 == 0:
      print('Step %d' % i)
  end_time = time.time()
  print('Time per step:', (end_time - start_time) / num_chain_iter)
  
with tf.Graph().as_default() as g:
  r_hat = tfp.mcmc.potential_scale_reduction(precision_samples)
  g.finalize()

with tf.Session(graph=g) as sess:
  print('r_hat:\n', sess.run(r_hat))

# The output samples have shape [n_steps, n_chains, 2, 2]
# Flatten them to [n_steps * n_chains, 2, 2] via reshape:
precision_samples_reshaped = np.reshape(precision_samples, newshape=[-1, 2, 2])

print('True posterior mean:\n', posterior_mean)
print('Mean of samples:\n', np.mean(precision_samples_reshaped, axis=0))

print('True posterior standard deviation:\n', posterior_sd)
print('Standard deviation of samples:\n', np.std(precision_samples_reshaped, axis=0))
