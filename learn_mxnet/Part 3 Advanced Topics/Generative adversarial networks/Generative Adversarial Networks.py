import matplotlib.pyplot 
import mxnet
import numpy
import datetime
import os
import time

ctx = mxnet.gpu()

X = mxnet.nd.random_normal(shape=(1000, 2))
A = mxnet.nd.array([[1, 2], [-0.1, 0.5]])
b = mxnet.nd.array([1, 2])
X = mxnet.nd.dot(X,A) + b
Y = mxnet.nd.ones(shape=(1000,1))
# and stick them into an iterator
batch_size = 4
train_data = mxnet.io.NDArrayIter(X, Y, batch_size, shuffle=True)

matplotlib.pyplot.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
matplotlib.pyplot.show()
print("The covariance matrix is")
print(mxnet.nd.dot(A, A.T))

# build the generator
netG = mxnet.gluon.nn.Sequential()
with netG.name_scope():
    netG.add(mxnet.gluon.nn.Dense(2))
# build the discriminator (with 5 and 3 hidden units respectively)
netD = mxnet.gluon.nn.Sequential()
with netD.name_scope():
    netD.add(mxnet.gluon.nn.Dense(5, activation='tanh'))
    netD.add(mxnet.gluon.nn.Dense(3 ,activation='tanh'))
    netD.add(mxnet.gluon.nn.Dense(2))
# loss
loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
# initialize the generator and the discriminator
netG.initialize(mxnet.init.Normal(0.02), ctx=ctx)
netD.initialize(mxnet.init.Normal(0.02), ctx=ctx)
# trainer for the generator and the discriminator
trainerG = mxnet.gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
trainerD = mxnet.gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

real_label = mxnet.nd.ones((batch_size,), ctx=ctx)
fake_label = mxnet.nd.zeros((batch_size,), ctx=ctx)
metric = mxnet.metric.Accuracy()
# set up logging

stamp =  datetime.datetime.now().strftime('%Y_%m_%d-%H_%M')
for epoch in range(10):
    tic = time.time()
    train_data.reset()
    for i, batch in enumerate(train_data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = batch.data[0].as_in_context(ctx)
        noise = mxnet.nd.random_normal(shape=(batch_size, 2), ctx=ctx)

        with mxnet.autograd.record():
            real_output = netD(data)
            errD_real = loss(real_output, real_label)

            fake = netG(noise)
            fake_output = netD(fake.detach())
            errD_fake = loss(fake_output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()

        trainerD.step(batch_size)
        metric.update([real_label,], [real_output,])
        metric.update([fake_label,], [fake_output,])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with mxnet.autograd.record():
            output = netD(fake)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch_size)

    name, acc = metric.get()
    metric.reset()
    print('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    print('time: %f' % (time.time() - tic))
    noise = mxnet.nd.random_normal(shape=(100, 2), ctx=ctx)
    fake = netG(noise)
    matplotlib.pyplot.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
    matplotlib.pyplot.scatter(fake[:,0].asnumpy(),fake[:,1].asnumpy())
    matplotlib.pyplot.show()

noise = mxnet.nd.random_normal(shape=(100, 2), ctx=ctx)
fake = netG(noise)
matplotlib.pyplot.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
matplotlib.pyplot.scatter(fake[:,0].asnumpy(),fake[:,1].asnumpy())
matplotlib.pyplot.show()