import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

context = mx.gpu(0)
args_data = '../data/nlp/ptb.'
args_model = 'lstm'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 10.0
args_clip = 0.2
args_epochs = 2
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500
args_save = 'model.param'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return mx.nd.array(ids, dtype='int32')
def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data
def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

corpus = Corpus(args_data)
ntokens = len(corpus.dictionary)
train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
test_data = batchify(corpus.test, args_batch_size).as_in_context(context)
num_batches = int(np.ceil( (train_data.shape[0] - 1)/args_bptt) )

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""
    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu',
                                   dropout=dropout, input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden
    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
    def set_params_to(self, new_values):
        for model_param, new_value in zip(self.collect_params().values(), new_values):
            model_param_ctx = model_param.list_ctx()[0]
            model_param._data[ model_param_ctx ] = new_value
        return
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

baseline_model = RNNModel(args_model, ntokens, args_emsize, args_nhid, args_nlayers, args_dropout, args_tied)
baseline_model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(
    baseline_model.collect_params(), 'sgd',
    {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
smce_loss = gluon.loss.SoftmaxCrossEntropyLoss()    

def train_baseline(model):
    global args_lr
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = smce_loss(output, target)
                L.backward()
            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)
            trainer.step(args_batch_size * args_bptt)
            total_L += mx.nd.sum(L).asscalar()
            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0
        val_L = evaluate(val_data, model)
        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))
        if val_L < best_val:
            best_val = val_L
            test_L = evaluate(test_data, model)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd', {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
            model.load_params(args_save, context)
    return
def evaluate(data_source, model):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = smce_loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

train_baseline(baseline_model)
baseline_model.load_params(args_save, context)
test_L = evaluate(test_data, baseline_model)
print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))

class ScaleMixturePrior(object):
    def __init__(self, alpha, sigma1, sigma2):
        self.alpha = mx.nd.array([alpha], ctx=context)
        self.one_minus_alpha = mx.nd.array([1 - alpha], ctx=context)
        self.zero = mx.nd.array([0.0], ctx=context)
        self.sigma1 = mx.nd.array([sigma1], ctx=context)
        self.sigma2 = mx.nd.array([sigma2], ctx=context)
        return
    def log_prob(self, model_params):
        total_log_prob = None
        for i, model_param in enumerate(model_params):
            p1 = gaussian_prob(model_param, self.zero, self.sigma1)
            p2 = gaussian_prob(model_param, self.zero, self.sigma2)
            log_prob = mx.nd.sum(mx.nd.log(self.alpha * p1 + self.one_minus_alpha * p2))
            if i == 0: total_log_prob = log_prob
            else: total_log_prob = total_log_prob + log_prob
        return total_log_prob
# Define some auxiliary functions
def log_gaussian_prob(x, mu, sigma):
    return - mx.nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
def gaussian_prob(x, mu, sigma):
    scaling = 1.0 / mx.nd.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = mx.nd.exp(-(x - mu)**2 / (2.0 * sigma ** 2))
    return scaling * bell

class VariationalPosterior(object):
    def __init__(self, model, var_mu_init_scale, var_sigma_init_scale):
        self.var_mus = []
        self.var_rhos = []
        self.raw_var_mus = []
        self.raw_var_rhos = []
        var_rho_init_scale = inv_softplus(var_sigma_init_scale)
        for i, model_param in enumerate(model.collect_params().values()):
            var_mu = gluon.Parameter('var_mu_{}'.format(i), shape=model_param.shape,
                init=mx.init.Normal(var_mu_init_scale))
            var_mu.initialize(ctx=context)
            self.var_mus.append(var_mu)
            self.raw_var_mus.append(var_mu.data(context))
            var_rho = gluon.Parameter(
                'var_rho_{}'.format(i), shape=model_param.shape,
                init=mx.init.Constant(var_rho_init_scale))
            var_rho.initialize(ctx=context)
            self.var_rhos.append(var_rho)
            self.raw_var_rhos.append(var_rho.data(context))
        self.var_params = self.var_mus + self.var_rhos
        return
    def log_prob(self, model_params):
        log_probs = [
            mx.nd.sum(log_gaussian_prob(model_param, raw_var_mu, softplus(raw_var_rho)))
            for (model_param, raw_var_mu, raw_var_rho)
            in zip(model_params, self.raw_var_mus, self.raw_var_rhos)]
        total_log_prob = log_probs[0]
        for log_prob in log_probs[1:]:
            total_log_prob = total_log_prob + log_prob
        return total_log_prob
    def sample_model_params(self):
        model_params = []
        for raw_var_mu, raw_var_rho in zip(self.raw_var_mus, self.raw_var_rhos):
            epsilon = mx.nd.random_normal(shape=raw_var_mu.shape, loc=0., scale=1.0, ctx=context)
            var_sigma = softplus(raw_var_rho)
            model_param = raw_var_mu + var_sigma * epsilon
            model_params.append(model_param)
        return model_params
    def num_params(self):
        return sum([
            2 * np.prod(param.shape)
            for param in self.var_mus])
# Define some auxiliary functions
def softplus(x):
    return mx.nd.log(1. + mx.nd.exp(x))
def inv_softplus(x):
    if x <= 0: raise ValueError("x must be > 0: {}".format(x))
    return np.log(np.exp(x) - 1.0)

class BBB_Loss(gluon.loss.Loss):
    def __init__(self, prior, var_posterior, log_likelihood, num_batches, weight=None, batch_axis=0, **kwargs):
        super(BBB_Loss, self).__init__(weight, batch_axis, **kwargs)
        self.prior = prior
        self.var_posterior = var_posterior
        self.log_likelihood = log_likelihood
        self.num_batches = num_batches
        return
    def forward(self, yhat, y, sampled_params, sample_weight=None):
        neg_log_likelihood = mx.nd.sum(self.log_likelihood(yhat, y))
        prior_log_prob = mx.nd.sum(self.prior.log_prob(sampled_params))
        var_post_log_prob = mx.nd.sum(self.var_posterior.log_prob(sampled_params))
        kl_loss = var_post_log_prob - prior_log_prob
        var_loss = neg_log_likelihood + kl_loss / self.num_batches
        return var_loss, neg_log_likelihood

def train_bbb(model):
    global args_lr
    global args_ess_multiplier
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            x, y = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                sampled_params = var_posterior.sample_model_params()
                model.set_params_to(sampled_params)
                yhat, hidden = model(x, hidden)
                var_loss, L = bbb_loss(yhat, y, sampled_params)
                var_loss.backward()
            grads = [var_mu.grad(context) for var_mu in var_posterior.var_mus]
            effective_batch_size = (args_bptt * args_batch_size) + (var_posterior.num_params() / num_batches)
            gluon.utils.clip_global_norm(grads, args_clip * effective_batch_size)
            trainer.step(args_clip * effective_batch_size)
            total_L += mx.nd.sum(L).asscalar()
            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0
        model.set_params_to(var_posterior.raw_var_mus)
        val_L = evaluate(val_data, model)
        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))
        if val_L < best_val:
            best_val = val_L
            model.set_params_to(var_posterior.raw_var_mus)
            test_L = evaluate(test_data, model)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd', {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
            model.load_params(args_save, context)
    return

bbb_model = RNNModel(args_model, ntokens, args_emsize, args_nhid, args_nlayers, dropout=0.0, tie_weights=args_tied)
bbb_model.collect_params().initialize(mx.init.Xavier(), ctx=context)
prior = ScaleMixturePrior(alpha = 0.75, sigma1 = 0.001, sigma2 = 0.75)
var_posterior = VariationalPosterior(bbb_model, var_mu_init_scale = 0.05, var_sigma_init_scale = 0.01)
bbb_loss = BBB_Loss(prior, var_posterior, gluon.loss.SoftmaxCrossEntropyLoss(), num_batches)
trainer = gluon.Trainer(var_posterior.var_params, 'sgd',
    { 'learning_rate': args_lr, 'momentum': 0, 'wd': 0 })

train_bbb(bbb_model)
bbb_model.load_params(args_save, context)
bbb_model.set_params_to(var_posterior.raw_var_mus)
test_L = evaluate(test_data, bbb_model)
print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))