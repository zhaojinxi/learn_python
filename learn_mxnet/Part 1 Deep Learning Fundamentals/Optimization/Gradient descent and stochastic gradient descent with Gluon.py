import mxnet
import numpy
import random

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
net = mxnet.gluon.nn.Sequential()
net.add(mxnet.gluon.nn.Dense(1))
square_loss = mxnet.gluon.loss.L2Loss()

#%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    net.collect_params().initialize(mxnet.init.Normal(sigma=1), force_reinit=True)
    # SGD.
    trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    data_iter = mxnet.gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    total_loss = [numpy.mean(square_loss(net(X), y).asnumpy())]
    for epoch in range(1, epochs + 1):
        # Decay learning rate.
        if epoch > 2:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (data, label) in enumerate(data_iter):
            with mxnet.autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % period == 0:
                total_loss.append(numpy.mean(square_loss(net(X), y).asnumpy()))
        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" %
              (batch_size, trainer.learning_rate, epoch, total_loss[-1]))

    print('w:', numpy.reshape(net[0].weight.data().asnumpy(), (1, -1)),
          'b:', net[0].bias.data().asnumpy()[0], '\n')
    x_axis = numpy.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

train(batch_size=1, lr=0.2, epochs=3, period=10)

train(batch_size=1000, lr=0.999, epochs=3, period=1000)

train(batch_size=10, lr=0.2, epochs=3, period=10)

train(batch_size=10, lr=5, epochs=3, period=10)

train(batch_size=10, lr=0.002, epochs=3, period=10)