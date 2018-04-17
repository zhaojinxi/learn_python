import numpy
import mxnet
import matplotlib.pyplot

data_ctx = mxnet.cpu()
model_ctx = mxnet.gpu()
# model_ctx = mxnet.gpu()

def transform(data, label):
    return data.astype(numpy.float32)/255, label.astype(numpy.float32)
mnist_train = mxnet.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mxnet.gluon.data.vision.MNIST(train=False, transform=transform)

image, label = mnist_train[0]
print(image.shape, label)

num_inputs = 784
num_outputs = 10
num_examples = 60000

im = mxnet.nd.tile(image, (1,1,3))
print(im.shape)

matplotlib.pyplot.imshow(im.asnumpy())
matplotlib.pyplot.show()

batch_size = 64
train_data = mxnet.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

test_data = mxnet.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

W = mxnet.nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
b = mxnet.nd.random_normal(shape=num_outputs,ctx=model_ctx)
params = [W, b]

for param in params:
    param.attach_grad()

def softmax(y_linear):
    exp = mxnet.nd.exp(y_linear-mxnet.nd.max(y_linear))
    norms = mxnet.nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / norms

sample_y_linear = mxnet.nd.random_normal(shape=(2,10))
sample_yhat = softmax(sample_y_linear)
print(sample_yhat)

print(mxnet.nd.sum(sample_yhat, axis=1))

def net(X):
    y_linear = mxnet.nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat

def cross_entropy(yhat, y):
    return - mxnet.nd.sum(y * mxnet.nd.log(yhat), axis=0, exclude=True)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = mxnet.nd.one_hot(label, 10)
        output = net(data)
        predictions = mxnet.nd.argmax(output, axis=1)
        numerator += mxnet.nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

evaluate_accuracy(test_data, net)

epochs = 10
learning_rate = .001
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = mxnet.nd.one_hot(label, 10)
        with mxnet.autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += mxnet.nd.sum(loss).asscalar()
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

# Define the function to do prediction
def model_predict(net,data):
    output = net(data)
    return mxnet.nd.argmax(output, axis=1)
# let's sample 10 random data points from the test set
sample_data = mxnet.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
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