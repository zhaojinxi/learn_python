import mxnet
import numpy
import matplotlib.pyplot

ctx = mxnet.cpu()
mxnet.random.seed(1)

# for plotting purposes
#%matplotlib inline

mnist = mxnet.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.ArrayDataset(mnist["train_data"][:num_examples], mnist["train_label"][:num_examples].astype(numpy.float32)), batch_size, shuffle=True)
test_data = mxnet.gluon.data.DataLoader(mxnet.gluon.data.ArrayDataset(mnist["test_data"][:num_examples], mnist["test_label"][:num_examples].astype(numpy.float32)), batch_size, shuffle=False)

W = mxnet.ndarray.random_normal(shape=(784,10))
b = mxnet.ndarray.random_normal(shape=10)
params = [W, b]
for param in params:
    param.attach_grad()
def net(X):
    y_linear = mxnet.ndarray.dot(X, W) + b
    yhat = mxnet.ndarray.softmax(y_linear, axis=1)
    return yhat

def cross_entropy(yhat, y):
    return - mxnet.ndarray.sum(y * mxnet.ndarray.log(yhat), axis=0, exclude=True)
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    loss_avg = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = mxnet.ndarray.one_hot(label, 10)
        output = net(data)
        loss = cross_entropy(output, label_one_hot)
        predictions = mxnet.ndarray.argmax(output, axis=1)
        numerator += mxnet.ndarray.sum(predictions == label)
        denominator += data.shape[0]
        loss_avg = loss_avg*i/(i+1) + mxnet.ndarray.mean(loss).asscalar()/(i+1)
    return (numerator / denominator).asscalar(), loss_avg

def plot_learningcurves(loss_tr,loss_ts, acc_tr,acc_ts):
    xs = list(range(len(loss_tr)))

    f = matplotlib.pyplot.figure(figsize=(12,6))
    fg1 = f.add_subplot(121)
    fg2 = f.add_subplot(122)

    fg1.set_xlabel('epoch',fontsize=14)
    fg1.set_title('Comparing loss functions')
    fg1.semilogy(xs, loss_tr)
    fg1.semilogy(xs, loss_ts)
    fg1.grid(True,which="both")

    fg1.legend(['training loss', 'testing loss'],fontsize=14)

    fg2.set_title('Comparing accuracy')
    fg1.set_xlabel('epoch',fontsize=14)
    fg2.plot(xs, acc_tr)
    fg2.plot(xs, acc_ts)
    fg2.grid(True,which="both")
    fg2.legend(['training accuracy', 'testing accuracy'],fontsize=14)
    matplotlib.pyplot.show()

epochs = 1000
moving_loss = 0.
niter=0
loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = mxnet.ndarray.one_hot(label, 10)
        with mxnet.autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, .001)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        moving_loss = .99 * moving_loss + .01 * mxnet.ndarray.mean(loss).asscalar()
        est_loss = moving_loss/(1-0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)
    if e % 100 == 99:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" % (e+1, train_loss, test_loss, train_accuracy, test_accuracy))
## Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)

def l2_penalty(params):
    penalty = mxnet.ndarray.zeros(shape=1)
    for param in params:
        penalty = penalty + mxnet.ndarray.sum(param ** 2)
    return penalty

for param in params:
    param[:] = mxnet.ndarray.random_normal(shape=param.shape)

epochs = 1000
moving_loss = 0.
l2_strength = .1
niter=0
loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = mxnet.ndarray.one_hot(label, 10)
        with mxnet.autograd.record():
            output = net(data)
            loss = mxnet.ndarray.sum(cross_entropy(output, label_one_hot)) + l2_strength * l2_penalty(params)
        loss.backward()
        SGD(params, .001)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        moving_loss = .99 * moving_loss + .01 * mxnet.ndarray.mean(loss).asscalar()
        est_loss = moving_loss/(1-0.99**niter)


    test_accuracy, test_loss = evaluate_accuracy(test_data, net)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)
    if e % 100 == 99:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

## Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)