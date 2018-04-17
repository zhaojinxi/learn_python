import matplotlib
import matplotlib.pyplot
import numpy
import mxnet
import random

# Mini-batch stochastic gradient descent.
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

mxnet.random.seed(1)
random.seed(1)

# Generate data.
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = mxnet.ndarray.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * mxnet.ndarray.random_normal(scale=1, shape=y.shape)
dataset = mxnet.gluon.data.ArrayDataset(X, y)
# Construct data iterator.
def data_iter(batch_size):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        j = mxnet.ndarray.array(idx[i: min(i + batch_size, num_examples)])
        yield batch_i, X.take(j), y.take(j)
# Initialize model parameters.
def init_params():
    w = mxnet.ndarray.random_normal(scale=1, shape=(num_inputs, 1))
    b = mxnet.ndarray.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params
# Linear regression.
def net(X, w, b):
    return mxnet.ndarray.dot(X, w) + b
# Loss function.
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

#%matplotlib inline
matplotlib.rcParams['figure.dpi']= 120
def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    w, b = init_params()
    total_loss = [numpy.mean(square_loss(net(X, w, b), y).asnumpy())]
    # Epoch starts from 1.
    for epoch in range(1, epochs + 1):
        # Decay learning rate.
        if epoch > 2:
            lr *= 0.1
        for batch_i, data, label in data_iter(batch_size):
            with mxnet.autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            sgd([w, b], lr, batch_size)
            if batch_i * batch_size % period == 0:
                total_loss.append(
                    numpy.mean(square_loss(net(X, w, b), y).asnumpy()))
        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" %
              (batch_size, lr, epoch, total_loss[-1]))
    print('w:', numpy.reshape(w.asnumpy(), (1, -1)),
          'b:', b.asnumpy()[0], '\n')
    x_axis = numpy.linspace(0, epochs, len(total_loss), endpoint=True)
    matplotlib.pyplot.semilogy(x_axis, total_loss)
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.show()

train(batch_size=1, lr=0.2, epochs=3, period=10)

train(batch_size=1000, lr=0.999, epochs=3, period=1000)

train(batch_size=10, lr=0.2, epochs=3, period=10)

train(batch_size=10, lr=5, epochs=3, period=10)

train(batch_size=10, lr=0.002, epochs=3, period=10)