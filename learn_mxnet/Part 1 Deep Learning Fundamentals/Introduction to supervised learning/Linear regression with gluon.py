import mxnet
import matplotlib.pyplot

data_ctx = mxnet.cpu()
model_ctx = mxnet.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
X = mxnet.nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * mxnet.nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

batch_size = 4
train_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)

net = mxnet.gluon.nn.Dense(1, in_units=2)

#print(net.weight)
#print(net.bias)

net.collect_params()

type(net.collect_params())

net.collect_params().initialize(mxnet.init.Normal(sigma=1.), ctx=model_ctx)

example_data = mxnet.nd.array([[4,7]])
net(example_data)

print(net.weight.data())
print(net.bias.data())

net = mxnet.gluon.nn.Dense(1)
net.collect_params().initialize(mxnet.init.Normal(sigma=1.), ctx=model_ctx)

square_loss = mxnet.gluon.loss.L2Loss()

trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})

epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size
for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with mxnet.autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += mxnet.nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)

# plot the convergence of the estimated loss function
matplotlib.pyplot.figure(num=None,figsize=(8, 6))
matplotlib.pyplot.plot(loss_sequence)
# Adding some bells and whistles to the plot
matplotlib.pyplot.grid(True, which="both")
matplotlib.pyplot.xlabel('epoch',fontsize=14)
matplotlib.pyplot.ylabel('average loss',fontsize=14)
matplotlib.pyplot.show()

params = net.collect_params() # this returns a ParameterDict
print('The type of "params" is a ',type(params))
# A ParameterDict is a dictionary of Parameter class objects
# therefore, here is how we can read off the parameters from it.
for param in params.values():
    print(param.name,param.data())