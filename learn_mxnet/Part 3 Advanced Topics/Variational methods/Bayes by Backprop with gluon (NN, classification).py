import collections
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from matplotlib import pyplot as plt
from mxnet import gluon
from functools import reduce
import operator

config = {
    "num_hidden_layers": 2,
    "num_hidden_units": 400, 
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "num_samples": 1,
    "pi": 0.25,
    "sigma_p": 1.0,
    "sigma_p1": 0.75,
    "sigma_p2": 0.01,
}

ctx = mx.gpu()

def transform(data, label):
    return data.astype(np.float32)/126.0, label.astype(np.float32)

mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = config['batch_size']
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)
num_train = sum([batch_size for i in train_data])
num_batches = num_train / batch_size

num_layers = config['num_hidden_layers']
num_hidden = config['num_hidden_units']
net = gluon.nn.Sequential()
with net.name_scope():
    for i in range(num_layers):
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

class BBBLoss(gluon.loss.Loss):
    def __init__(self, log_prior="gaussian", log_likelihood="softmax_cross_entropy", sigma_p1=1.0, sigma_p2=0.1, pi=0.5, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss, self).__init__(weight, batch_axis, **kwargs)
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.pi = pi
    def log_softmax_likelihood(self, yhat_linear, y):
        return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
    def log_gaussian(self, x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
    def gaussian_prior(self, x):
        sigma_p = nd.array([self.sigma_p1], ctx=ctx)
        return nd.sum(self.log_gaussian(x, 0., sigma_p))
    def gaussian(self, x, mu, sigma):
        scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell
    def scale_mixture_prior(self, x):
        sigma_p1 = nd.array([self.sigma_p1], ctx=ctx)
        sigma_p2 = nd.array([self.sigma_p2], ctx=ctx)
        pi = self.pi
        first_gaussian = pi * self.gaussian(x, 0., sigma_p1)
        second_gaussian = (1 - pi) * self.gaussian(x, 0., sigma_p2)
        return nd.log(first_gaussian + second_gaussian)
    def hybrid_forward(self, F, output, label, params, mus, sigmas, sample_weight=None):
        log_likelihood_sum = nd.sum(self.log_softmax_likelihood(output, label))
        prior = None
        if self.log_prior == "gaussian":
            prior = self.gaussian_prior
        elif self.log_prior == "scale_mixture":
            prior = self.scale_mixture_prior
        log_prior_sum = sum([nd.sum(prior(param)) for param in params])
        log_var_posterior_sum = sum([nd.sum(self.log_gaussian(params[i], mus[i], sigmas[i])) for i in range(len(params))])
        return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum
bbb_loss = BBBLoss(log_prior="scale_mixture", sigma_p1=config['sigma_p1'], sigma_p2=config['sigma_p2'])

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

for i, (data, label) in enumerate(train_data):
    data = data.as_in_context(ctx).reshape((-1, 784))
    net(data)
    break

weight_scale = .1
rho_offset = -3
# initialize variational parameters; mean and variance for each weight
mus = []
rhos = []
shapes = list(map(lambda x: x.shape, net.collect_params().values()))
for shape in shapes:
    mu = gluon.Parameter('mu', shape=shape, init=mx.init.Normal(weight_scale))
    rho = gluon.Parameter('rho',shape=shape, init=mx.init.Constant(rho_offset))
    mu.initialize(ctx=ctx)
    rho.initialize(ctx=ctx)
    mus.append(mu)
    rhos.append(rho)
variational_params = mus + rhos
raw_mus = list(map(lambda x: x.data(ctx), mus))
raw_rhos = list(map(lambda x: x.data(ctx), rhos))

trainer = gluon.Trainer(variational_params, 'adam', {'learning_rate': config['learning_rate']})

def sample_epsilons(param_shapes):
    epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]
    return epsilons

def softplus(x):
    return nd.log(1. + nd.exp(x))
def transform_rhos(rhos):
    return [softplus(rho) for rho in rhos]    

def transform_gaussian_samples(mus, sigmas, epsilons):
    samples = []
    for j in range(len(mus)):
        samples.append(mus[j] + sigmas[j] * epsilons[j])
    return samples

def generate_weight_sample(layer_param_shapes, mus, rhos):
    # sample epsilons from standard normal
    epsilons = sample_epsilons(layer_param_shapes)
    # compute softplus for variance
    sigmas = transform_rhos(rhos)
    # obtain a sample from q(w|theta) by transforming the epsilons
    layer_params = transform_gaussian_samples(mus, sigmas, epsilons)
    return layer_params, sigmas

def evaluate_accuracy(data_iterator, net, layer_params):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        for l_param, param in zip(layer_params, net.collect_params().values()):
            param._data[list(param._data.keys())[0]] = l_param
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = config['epochs']
learning_rate = config['learning_rate']
smoothing_constant = .01
train_acc = []
test_acc = []
for e in range(epochs):
    for i, (data, label) in enumerate(train_data): 
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            # generate sample
            layer_params, sigmas = generate_weight_sample(shapes, raw_mus, raw_rhos)
            # overwrite network parameters with sampled parameters 
            for sample, param in zip(layer_params, net.collect_params().values()):
                param._data[list(param._data.keys())[0]] = sample 
            # forward-propagate the batch 
            output = net(data)
            # calculate the loss
            loss = bbb_loss(output, label_one_hot, layer_params, raw_mus, sigmas)
            # backpropagate for gradient calculation
            loss.backward()
        trainer.step(data.shape[0])
        # calculate moving loss for monitoring convergence
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
    test_accuracy = evaluate_accuracy(test_data, net, raw_mus)
    train_accuracy = evaluate_accuracy(train_data, net, raw_mus)
    train_acc.append(np.asscalar(train_accuracy))
    test_acc.append(np.asscalar(test_accuracy))
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))
plt.plot(train_acc)
plt.plot(test_acc)
plt.show()

def gaussian(x, mu, sigma):
    scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return scaling * bell
def show_weight_dist(mean, variance):
    sigma = nd.sqrt(variance)
    x = np.linspace(mean.asscalar() - 4*sigma.asscalar(), mean.asscalar() + 4*sigma.asscalar(), 100)
    plt.plot(x, gaussian(nd.array(x, ctx=ctx), mean, sigma).asnumpy())
    plt.show()
mu = raw_mus[0][0][0]
var = softplus(raw_rhos[0][0][0]) ** 2
show_weight_dist(mu, var)

def signal_to_noise_ratio(mus, sigmas):
    sign_to_noise = []
    for j in range(len(mus)):
        sign_to_noise.extend([nd.abs(mus[j]) / sigmas[j]])
    return sign_to_noise

def vectorize_matrices_in_vector(vec):
    for i in range(0, (num_layers + 1) * 2, 2):
        if i == 0:
            vec[i] = nd.reshape(vec[i], num_inputs * num_hidden)
        elif i == num_layers * 2:
            vec[i] = nd.reshape(vec[i], num_hidden * num_outputs)
        else:
            vec[i] = nd.reshape(vec[i], num_hidden * num_hidden)   
    return vec
def concact_vectors_in_vector(vec):
    concat_vec = vec[0]
    for i in range(1, len(vec)):
        concat_vec = nd.concat(concat_vec, vec[i], dim=0)
    return concat_vec
def transform_vector_structure(vec):
    vec = vectorize_matrices_in_vector(vec)
    vec = concact_vectors_in_vector(vec)
    return vec

def prod(iterable):
    return reduce(operator.mul, iterable, 1)
def restore_weight_structure(vec):
    pruned_weights = []
    index = 0
    for shape in shapes:
        incr = prod(shape)
        pruned_weights.extend([nd.reshape(vec[index : index + incr], shape)])
        index += incr
    return pruned_weights

def prune_weights(sign_to_noise_vec, prediction_vector, percentages):
    pruning_indices = nd.argsort(sign_to_noise_vec, axis=0)
    for percentage in percentages:
        prediction_vector = mus_copy_vec.copy()
        pruning_indices_percent = pruning_indices[0:int(len(pruning_indices)*percentage)]
        for pr_ind in pruning_indices_percent:
            prediction_vector[int(pr_ind.asscalar())] = 0
        pruned_weights = restore_weight_structure(prediction_vector)
        test_accuracy = evaluate_accuracy(test_data, net, pruned_weights)
        print("%s --> %s" % (percentage, test_accuracy))

sign_to_noise = signal_to_noise_ratio(raw_mus, sigmas)
sign_to_noise_vec = transform_vector_structure(sign_to_noise)
mus_copy = raw_mus.copy()
mus_copy_vec = transform_vector_structure(mus_copy)
prune_weights(sign_to_noise_vec, mus_copy_vec, [0.1, 0.25, 0.5, 0.75, 0.95, 0.98, 1.0])