import mxnet
import os

ctx = mxnet.gpu()

X = mxnet.nd.ones((100, 100))
Y = mxnet.nd.zeros((100, 100))
os.makedirs('checkpoints', exist_ok=True)
filename = "checkpoints/test1.params"
mxnet.nd.save(filename, [X, Y])

A, B = mxnet.nd.load(filename)
print(A)
print(B)

mydict = {"X": X, "Y": Y}
filename = "checkpoints/test2.params"
mxnet.nd.save(filename, mydict)

C = mxnet.nd.load(filename)
print(C)

num_hidden = 256
num_outputs = 1
net = mxnet.gluon.nn.Sequential()
with net.name_scope():
    net.add(mxnet.gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(mxnet.gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(mxnet.gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mxnet.init.Normal(sigma=1.), ctx=ctx)
net(mxnet.nd.ones((1, 100), ctx=ctx))

filename = "checkpoints/testnet.params"
net.save_params(filename)
net2 = mxnet.gluon.nn.Sequential()
with net2.name_scope():
    net2.add(mxnet.gluon.nn.Dense(num_hidden, activation="relu"))
    net2.add(mxnet.gluon.nn.Dense(num_hidden, activation="relu"))
    net2.add(mxnet.gluon.nn.Dense(num_outputs))
net2.load_params(filename, ctx=ctx)
net2(mxnet.nd.ones((1, 100), ctx=ctx))