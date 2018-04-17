import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block

###########################
#  Specify the context we'll be using
###########################
ctx = mx.gpu()
###########################
#  Load up our dataset
###########################
batch_size = 64
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(root='../../data/mnist/', train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(root='../../data/mnist/', train=False, transform=transform), batch_size, shuffle=False)

net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Dense(128, activation="relu"))
    net1.add(gluon.nn.Dense(64, activation="relu"))
    net1.add(gluon.nn.Dense(10))

class MLP(Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(128)
            self.dense1 = nn.Dense(64)
            self.dense2 = nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense1(x))
        return self.dense2(x)

net2 = MLP()

net2.initialize(ctx=ctx)

for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
net2(data[0:1])

net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Dense(128, activation="relu"))
    net1.add(gluon.nn.Dense(64, activation="relu"))
    net1.add(gluon.nn.Dense(10))

def forward(self, x):
    for block in self._children:
        x = block(x)
    return x

print(net1.collect_params())

net1.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

net1(data)
print(net1.collect_params())

net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(128, in_units=784, activation="relu"))
    net2.add(gluon.nn.Dense(64, in_units=128, activation="relu"))
    net2.add(gluon.nn.Dense(10, in_units=64))

net2.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
print(net2.collect_params())