import mxnet
import numpy
import matplotlib.pyplot

data_ctx = mxnet.cpu()
model_ctx = mxnet.gpu()
# model_ctx = mxnet.gpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
def transform(data, label):
    return data.astype(numpy.float32)/255, label.astype(numpy.float32)
train_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

net = mxnet.gluon.nn.Dense(num_outputs)

net.collect_params().initialize(mxnet.init.Normal(sigma=1.), ctx=model_ctx)

softmax_cross_entropy = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()

trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

def evaluate_accuracy(data_iterator, net):
    acc = mxnet.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = mxnet.nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

evaluate_accuracy(test_data, net)

epochs = 10
moving_loss = 0.
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        with mxnet.autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += mxnet.nd.sum(loss).asscalar()
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

def model_predict(net,data):
    output = net(data.as_in_context(model_ctx))
    return mxnet.nd.argmax(output, axis=1)
# let's sample 10 random data points from the test set
sample_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.vision.MNIST(train=False, transform=transform), 10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = mxnet.nd.transpose(data,(1,0,2,3))
    im = mxnet.nd.reshape(im,(28,10*28,1))
    imtiles = mxnet.nd.tile(im, (1,1,3))
    matplotlib.pyplot.imshow(imtiles.asnumpy())
    matplotlib.pyplot.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    break