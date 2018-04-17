import mxnet
import matplotlib.pyplot

mxnet.random.seed(1)

data_ctx = mxnet.cpu()
model_ctx = mxnet.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
X = mxnet.nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * mxnet.nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise

print(X[0])
print(y[0])

print(2 * X[0, 0] - 3.4 * X[0, 1] + 4.2)

matplotlib.pyplot.scatter(X[:, 1].asnumpy(),y.asnumpy())
matplotlib.pyplot.show()

batch_size = 4
train_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)

for i, (data, label) in enumerate(train_data):
    print(data, label)
    break

for i, (data, label) in enumerate(train_data):
    print(data, label)
    break

counter = 0
for i, (data, label) in enumerate(train_data):
    pass
print(i+1)

w = mxnet.nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = mxnet.nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [w, b]

for param in params:
    param.attach_grad()

def net(X):
    return mxnet.nd.dot(X, w) + b

def square_loss(yhat, y):
    return mxnet.nd.mean((yhat - y) ** 2)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

epochs = 10
learning_rate = .0001
num_batches = num_examples/batch_size
for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with mxnet.autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    print(cumulative_loss / num_batches)

############################################
#    Re-initialize parameters because they
#    were already trained in the first loop
############################################
w[:] = mxnet.nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b[:] = mxnet.nd.random_normal(shape=num_outputs, ctx=model_ctx)
############################################
#    Script to plot the losses over time
############################################
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = matplotlib.pyplot.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()

    matplotlib.pyplot.show()
learning_rate = .0001
losses = []
plot(losses, X)
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with mxnet.autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()

    print("Epoch %s, batch %s. Mean loss: %s" % (e, i, cumulative_loss/num_batches))
    losses.append(cumulative_loss/num_batches)
plot(losses, X)