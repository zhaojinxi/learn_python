import mxnet as mx
import numpy as np
from mxnet import nd, autograd
mx.random.seed(1)
ctx = mx.gpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../../data/mnist/', train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST('../../data/mnist/', train=False, transform=transform), batch_size, shuffle=False)

def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')
    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta
    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0, 2, 3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
    return out

A = nd.array([1,7,5,4,6,10], ctx=ctx).reshape((3,2))
A

pure_batch_norm(A,
    gamma = nd.array([1,1], ctx=ctx),
    beta=nd.array([0,0], ctx=ctx))

ga = nd.array([1,1], ctx=ctx)
be = nd.array([0,0], ctx=ctx)
B = nd.array([1,6,5,7,4,3,2,5,6,3,2,4,5,3,2,5,6], ctx=ctx).reshape((2,2,2,2))
B

pure_batch_norm(B, ga, be)

def batch_norm(X, gamma, beta, momentum = 0.9, eps = 1e-5, scope_name = '', is_training = True, debug = False):
    """compute the batch norm """
    global _BN_MOVING_MEANS, _BN_MOVING_VARS

    #########################
    # the usual batch norm transformation
    #########################

    if len(X.shape) not in (2, 4):
        raise ValueError('the input data shape should be one of:\n' + 'dense: (batch size, # of features)\n' + '2d conv: (batch size, # of features, height, width)')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        if is_training:
            # while training, we normalize the data using its mean and variance
            X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[scope_name]) *1.0 / nd.sqrt(_BN_MOVING_VARS[scope_name] + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0,2,3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        if is_training:
            # while training, we normalize the data using its mean and variance
            X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[scope_name].reshape((1, C, 1, 1))) * 1.0 \
                / nd.sqrt(_BN_MOVING_VARS[scope_name].reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    #########################
    # to keep the moving statistics
    #########################

    # init the attributes
    try: # to access them
        _BN_MOVING_MEANS, _BN_MOVING_VARS
    except: # error, create them
        _BN_MOVING_MEANS, _BN_MOVING_VARS = {}, {}

    # store the moving statistics by their scope_names, inplace
    if scope_name not in _BN_MOVING_MEANS:
        _BN_MOVING_MEANS[scope_name] = mean
    else:
        _BN_MOVING_MEANS[scope_name] = _BN_MOVING_MEANS[scope_name] * momentum + mean * (1.0 - momentum)
    if scope_name not in _BN_MOVING_VARS:
        _BN_MOVING_VARS[scope_name] = variance
    else:
        _BN_MOVING_VARS[scope_name] = _BN_MOVING_VARS[scope_name] * momentum + variance * (1.0 - momentum)

    #########################
    # debug info
    #########################
    if debug:
        print('== info start ==')
        print('scope_name = {}'.format(scope_name))
        print('mean = {}'.format(mean))
        print('var = {}'.format(variance))
        print('_BN_MOVING_MEANS = {}'.format(_BN_MOVING_MEANS[scope_name]))
        print('_BN_MOVING_VARS = {}'.format(_BN_MOVING_VARS[scope_name]))
        print('output = {}'.format(out))
        print('== info end ==')

    #########################
    # return
    #########################
    return out

#######################
#  Set the scale for weight initialization and choose
#  the number of hidden units in the fully-connected layer
#######################
weight_scale = .01
num_fc = 128
W1 = nd.random_normal(shape=(20, 1, 3,3), scale=weight_scale, ctx=ctx)
b1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)
gamma1 = nd.random_normal(shape=20, loc=1, scale=weight_scale, ctx=ctx)
beta1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)
W2 = nd.random_normal(shape=(50, 20, 5, 5), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)
gamma2 = nd.random_normal(shape=50, loc=1, scale=weight_scale, ctx=ctx)
beta2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)
W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)
gamma3 = nd.random_normal(shape=num_fc, loc=1, scale=weight_scale, ctx=ctx)
beta3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)
W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=10, scale=weight_scale, ctx=ctx)
params = [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3, gamma3, beta3, W4, b4]

for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)

def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

def net(X, is_training = True, debug=False):
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
    h1_normed = batch_norm(h1_conv, gamma1, beta1, scope_name='bn1', is_training=is_training)
    h1_activation = relu(h1_normed)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5), num_filter=50)
    h2_normed = batch_norm(h2_conv, gamma2, beta2, scope_name='bn2', is_training=is_training)
    h2_activation = relu(h2_normed)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))

    ########################
    #  Flattening h2 so that we can feed it into a fully-connected layer
    ########################
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))

    ########################
    #  Define the computation of the third (fully-connected) layer
    ########################
    h3_linear = nd.dot(h2, W3) + b3
    h3_normed = batch_norm(h3_linear, gamma3, beta3, scope_name='bn3', is_training=is_training)
    h3 = relu(h3_normed)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))

    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

    return yhat_linear

for data, _ in train_data:
    data = data.as_in_context(ctx)
    break

output = net(data, is_training=True, debug=True)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data, is_training=False) # attention here!
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = 1
moving_loss = 0.
learning_rate = .001
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            # we are in training process,
            # so we normalize the data using batch mean and variance
            output = net(data, is_training=True)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))