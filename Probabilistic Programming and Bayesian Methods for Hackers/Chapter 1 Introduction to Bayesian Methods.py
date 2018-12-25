#@title Imports and Global Variables (make sure to run this cell)  { display-mode: "form" }

#@markdown This sets the warning status (default is `ignore`, since this notebook runs correctly)
warning_status = "ignore"  #@param ["ignore", "always", "module", "once", "default", "error"]
import warnings
warnings.filterwarnings(warning_status)
with warnings.catch_warnings():
    warnings.filterwarnings(warning_status, category=DeprecationWarning)
    warnings.filterwarnings(warning_status, category=UserWarning)
import numpy as np
import os
#@markdown This sets the styles of the plotting (default is styled like plots from [FiveThirtyeight.com](https://fivethirtyeight.com/))
matplotlib_style = 'fivethirtyeight'  #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
import seaborn as sns; sns.set_context('notebook')
import tensorflow as tf
tfe = tf.contrib.eager

# Eager Execution
#@markdown Check the box below if you want to use [Eager Execution](https://www.tensorflow.org/guide/eager)
#@markdown Eager execution provides An intuitive interface, Easier debugging, and a control flow comparable to Numpy. You can read more about it on the [Google AI Blog](https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html)
use_tf_eager = True #@param {type:"boolean"}

# Use try/except so we can easily re-execute the whole notebook.
if use_tf_eager:
    try:
        tf.enable_eager_execution()
    except:
        pass

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

  
def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
      `namedtuple` or combinations thereof.
 
    Returns:
      ndarrays: Object with same structure as `tensors` except with `Tensor` or
        `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(tensors, [t.numpy() if tf.contrib.framework.is_tensor(t) else t for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def session_options(enable_gpu_ram_resizing=True, enable_xla=False):
    """
    Allowing the notebook to make use of GPUs if they're available.
    
    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear 
    algebra that optimizes TensorFlow computations.
    """
    config = tf.ConfigProto()
    config.log_device_placement = True
    if enable_gpu_ram_resizing:
        # `allow_growth=True` makes it possible to connect multiple colabs to your
        # GPU. Otherwise the colab malloc's all GPU ram.
        config.gpu_options.allow_growth = True
    if enable_xla:
        # Enable on XLA. https://www.tensorflow.org/performance/xla/.
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
    return config


def reset_sess(config=None):
    """
    Convenience function to create the TF graph & session or reset them.
    """
    if config is None:
        config = session_options(enable_gpu_ram_resizing=True, enable_xla=False)
    global sess
    tf.reset_default_graph()
    try:
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession(config=config)

reset_sess()

# Example: Mandatory coin-flip example

# Build graph.
probs_of_heads = tf.linspace(start=0., stop=1., num=100, name="linspace")
n_trials_ = [0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000]
coin_flip_prior = tfp.distributions.Bernoulli(probs=0.5)
coin_flip_data = coin_flip_prior.sample(n_trials_[-1])

n_trials_unpacked = tf.unstack(tf.constant(n_trials_))
counted = []  # this will be the list of processed head count tensors
for k, N in enumerate(n_trials_):
    result_tensor = tf.reduce_sum(coin_flip_data[:N])
    counted.append(result_tensor)

headcounts = tf.stack(counted, 0)

observed_head_probs = []  # this will be the list of processed probability tensors
for k, N in enumerate(n_trials_):
    result_tensor = tfp.distributions.Beta(concentration1 = tf.to_float(1 + headcounts[k]), concentration0 = tf.to_float(1 + n_trials_[k] - headcounts[k])).prob(probs_of_heads)
    observed_head_probs.append(result_tensor)

observed_probs_heads = tf.stack(observed_head_probs, 0)

# Execute graph
[
    n_trials_unpacked_,
    coin_flip_data_,
    probs_of_heads_,
    headcounts_,
    observed_probs_heads_,
] = evaluate([
    n_trials_unpacked,
    coin_flip_data,
    probs_of_heads,
    headcounts,
    observed_probs_heads,
])

# For the already prepared, I'm using Binomial's conj. prior.
plt.figure(figsize=(16, 9))
for i in range(len(n_trials_)):
    sx = plt.subplot(len(n_trials_)/2, 2, i+1)
    plt.xlabel("$p$, probability of heads") if i in [0, len(n_trials_)-1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    plt.plot(probs_of_heads_, observed_probs_heads_[i],  label="observe %d tosses,\n %d heads" % (n_trials_[i], headcounts_[i]))
    plt.fill_between(probs_of_heads_, 0, observed_probs_heads_[i],  color=TFColor[3], alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)
plt.suptitle("Bayesian updating of posterior probabilities", y=1.02, fontsize=14)
plt.tight_layout()

# Example: Bug, or just sweet, unintended feature?

# Defining our range of probabilities
p = tf.linspace(start=0., stop=1., num=50)

# Convert from TF to numpy.
[p_] = evaluate([p])

# Visualization.
plt.figure(figsize=(12.5, 6))
plt.plot(p_, 2*p_/(1+p_), color=TFColor[3], lw=3)
#plt.fill_between(p, 2*p/(1+p), alpha=.5, facecolor=["#A60628"])
plt.scatter(0.2, 2*(0.2)/1.2, s=140, c=TFColor[3])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r"Prior, $P(A) = p$")
plt.ylabel(r"Posterior, $P(A|X)$, with $P(A) = p$")
plt.title(r"Are there bugs in my code?");