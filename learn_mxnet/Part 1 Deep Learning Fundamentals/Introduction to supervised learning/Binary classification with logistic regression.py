import mxnet
import matplotlib.pyplot

def logistic(z):
    return 1. / (1. + mxnet.nd.exp(-z))
x = mxnet.nd.arange(-5, 5, .1)
y = logistic(x)
matplotlib.pyplot.plot(x.asnumpy(),y.asnumpy())
matplotlib.pyplot.show()

data_ctx = mxnet.gpu()
# Change this to `mxnet.gpu(0) if you would like to train on an NVIDIA GPU
model_ctx = mxnet.gpu()
with open("../../data/adult/a1a.train") as f:
    train_raw = f.read()
with open("../../data/adult/a1a.test") as f:
    test_raw = f.read()

def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123
    X = mxnet.nd.zeros((num_examples, num_features), ctx=data_ctx)
    Y = mxnet.nd.zeros((num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(train_lines):
        tokens = line.split()
        label = (int(tokens[0]) + 1) / 2  # Change label from {-1,1} to {0,1}
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y

Xtrain, Ytrain = process_data(train_raw)
Xtest, Ytest = process_data(test_raw)

print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)

print(mxnet.nd.sum(Ytrain)/len(Ytrain))
print(mxnet.nd.sum(Ytest)/len(Ytest))

batch_size = 64
train_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.ArrayDataset(Xtrain, Ytrain), batch_size=batch_size, shuffle=True)
test_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.ArrayDataset(Xtest, Ytest), batch_size=batch_size, shuffle=True)

net = mxnet.gluon.nn.Dense(1)
net.collect_params().initialize(mxnet.init.Normal(sigma=1.), ctx=model_ctx)

trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

def log_loss(output, y):
    yhat = logistic(output)
    return  - mxnet.nd.sum(  y * mxnet.nd.log(yhat) + (1-y) * mxnet.nd.log(1-yhat))

epochs = 30
loss_sequence = []
num_examples = len(Xtrain)
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with mxnet.autograd.record():
            output = net(data)
            loss = log_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += mxnet.nd.sum(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss ))
    loss_sequence.append(cumulative_loss)

# plot the convergence of the estimated loss function
#%matplotlib inline
matplotlib.pyplot.figure(num=None,figsize=(8, 6))
matplotlib.pyplot.plot(loss_sequence)
# Adding some bells and whistles to the plot
matplotlib.pyplot.grid(True, which="both")
matplotlib.pyplot.xlabel('epoch',fontsize=14)
matplotlib.pyplot.ylabel('average loss',fontsize=14)
matplotlib.pyplot.show()

num_correct = 0.0
num_total = len(Xtest)
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    output = net(data)
    prediction = (mxnet.nd.sign(output) + 1) / 2
    num_correct += mxnet.nd.sum(prediction == label)
print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar()/num_total, num_correct.asscalar(), num_total))