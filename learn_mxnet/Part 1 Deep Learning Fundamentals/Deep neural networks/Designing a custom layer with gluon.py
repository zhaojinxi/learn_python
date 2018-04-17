import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
mx.random.seed(1)

###########################
#  Speficy the context we'll be using
###########################
ctx = mx.cpu()
###########################
#  Load up our dataset
###########################
batch_size = 64
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(root='../../data/mnist/', train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(root='../../data/mnist/', train=False, transform=transform), batch_size, shuffle=False)

class CenteredLayer(Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - nd.mean(x)

net = CenteredLayer()
net(nd.array([1,2,3,4,5]))

net2 = nn.Sequential()
net2.add(nn.Dense(128))
net2.add(nn.Dense(10))
net2.add(CenteredLayer())

net2.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
output = net2(data[0:1])
print(output)

nd.mean(output)

my_param = gluon.Parameter("exciting_parameter_yay", grad_req='write', shape=(5,5))
print(my_param)

my_param.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
print(my_param.data())

# my_param = gluon.Parameter("exciting_parameter_yay", grad_req='write', shape=(5,5))
# my_param.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.gpu(0), mx.gpu(1)])
# print(my_param.data(mx.gpu(0)), my_param.data(mx.gpu(1)))

pd = gluon.ParameterDict(prefix="block1_")

pd.get("exciting_parameter_yay", grad_req='write', shape=(5,5))

pd["block1_exciting_parameter_yay"]

def relu(X):
    return nd.maximum(X, 0)

class MyDense(Block):
    ####################
    # We add arguments to our constructor (__init__)
    # to indicate the number of input units (``in_units``)
    # and output units (``units``)
    ####################
    def __init__(self, units, in_units=0, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.units = units
            self._in_units = in_units
            #################
            # We add the required parameters to the ``Block``'s ParameterDict ,
            # indicating the desired shape
            #################
            self.weight = self.params.get(
                'weight', init=mx.init.Xavier(magnitude=2.24),
                shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    #################
    #  Now we just have to write the forward pass.
    #  We could rely upong the FullyConnected primitative in NDArray,
    #  but it's better to get our hands dirty and write it out
    #  so you'll know how to compose arbitrary functions
    #################
    def forward(self, x):
        with x.context:
            linear = nd.dot(x, self.weight.data()) + self.bias.data()
            activation = relu(linear)
            return activation

dense = MyDense(20, in_units=10)
dense.collect_params().initialize(ctx=ctx)

dense.params

dense(nd.ones(shape=(2,10)))

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(MyDense(128, in_units=784))
    net.add(MyDense(64, in_units=128))
    net.add(MyDense(10, in_units=64))

net.collect_params().initialize(ctx=ctx)

loss = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

metric = mx.metric.Accuracy()
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        with autograd.record():
            data = data.as_in_context(ctx).reshape((-1,784))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)
            output = net(data)

        metric.update([label], [output])
    return metric.get()[1]

epochs = 2  # Low number for testing, set higher when you run!
moving_loss = 0.
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
            cross_entropy.backward()
        trainer.step(data.shape[0])
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Train_acc %s, Test_acc %s" % (e, train_accuracy, test_accuracy))