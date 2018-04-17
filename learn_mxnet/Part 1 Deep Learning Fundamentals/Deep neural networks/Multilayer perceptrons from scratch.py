import mxnet
import numpy

data_ctx = mxnet.cpu()
model_ctx = mxnet.gpu()
# model_ctx = mxnet.gpu(1)

num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
def transform(data, label):
    return data.astype(numpy.float32)/255, label.astype(numpy.float32)
train_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

#######################
#  Set some constants so it's easy to modify the network later
#######################
num_hidden = 256
weight_scale = .01
#######################
#  Allocate parameters for the first hidden layer
#######################
W1 = mxnet.nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = mxnet.nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)
#######################
#  Allocate parameters for the second hidden layer
#######################
W2 = mxnet.nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = mxnet.nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)
#######################
#  Allocate parameters for the output layer
#######################
W3 = mxnet.nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = mxnet.nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)
params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

def relu(X):
    return mxnet.nd.maximum(X, mxnet.nd.zeros_like(X))

def softmax(y_linear):
    exp = mxnet.nd.exp(y_linear-mxnet.nd.max(y_linear))
    partition = mxnet.nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition

def cross_entropy(yhat, y):
    return - mxnet.nd.nansum(y * mxnet.nd.log(yhat), axis=0, exclude=True)

def softmax_cross_entropy(yhat_linear, y):
    return - mxnet.nd.nansum(y * mxnet.nd.log_softmax(yhat_linear), axis=0, exclude=True)

def net(X):
    #######################
    #  Compute the first hidden layer
    #######################
    h1_linear = mxnet.nd.dot(X, W1) + b1
    h1 = relu(h1_linear)

    #######################
    #  Compute the second hidden layer
    #######################
    h2_linear = mxnet.nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)

    #######################
    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    #######################
    yhat_linear = mxnet.nd.dot(h2, W3) + b3
    return yhat_linear

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = mxnet.nd.argmax(output, axis=1)
        numerator += mxnet.nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = 10
learning_rate = .001
smoothing_constant = .01
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = mxnet.nd.one_hot(label, 10)
        with mxnet.autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += mxnet.nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))