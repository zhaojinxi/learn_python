import mxnet
import matplotlib.pyplot
import numpy
import random
import math

probabilities = mxnet.nd.ones(6) / 6
mxnet.nd.sample_multinomial(probabilities)

print(mxnet.nd.sample_multinomial(probabilities, shape=(10)))
print(mxnet.nd.sample_multinomial(probabilities, shape=(5,10)))

rolls = mxnet.nd.sample_multinomial(probabilities, shape=(1000))

counts = mxnet.nd.zeros((6,1000))
totals = mxnet.nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals

totals / 1000

counts

x = mxnet.nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])

matplotlib.pyplot.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
matplotlib.pyplot.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
matplotlib.pyplot.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
matplotlib.pyplot.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
matplotlib.pyplot.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
matplotlib.pyplot.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")
matplotlib.pyplot.axhline(y=0.16666, color='black', linestyle='dashed')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

# we go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (mxnet.nd.floor(data/128)).astype(numpy.float32), label.astype(numpy.float32)
mnist_train = mxnet.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mxnet.gluon.data.vision.MNIST(train=False, transform=transform)
# Initialize the count statistics for p(y) and p(x_i|y)
# We initialize all numbers with a count of 1 to ensure that we don't get a
# division by zero.  Statisticians call this Laplace smoothing.
ycount = mxnet.nd.ones(shape=(10))
xcount = mxnet.nd.ones(shape=(784, 10))
# Aggregate count statistics of how frequently a pixel is on (or off) for
# zeros and ones.
for data, label in mnist_train:
    x = data.reshape((784,))
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += x
# normalize the probabilities p(x_i|y) (divide per pixel counts by total
# count)
for i in range(10):
    xcount[:, i] = xcount[:, i]/ycount[i]
# likewise, compute the probability p(y)
py = ycount / mxnet.nd.sum(ycount)

fig, figarr = matplotlib.pyplot.subplots(1, 10, figsize=(15, 15))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)
matplotlib.pyplot.show()
print(py)

logxcount = mxnet.nd.log(xcount)
logxcountneg = mxnet.nd.log(1-xcount)
logpy = mxnet.nd.log(py)
fig, figarr = matplotlib.pyplot.subplots(2, 10, figsize=(15, 3))
# show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,))
    y = int(label)

    # we need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpx = logpy.copy()
    for i in range(10):
        # compute the log probability for a digit
        logpx[i] += mxnet.nd.dot(logxcount[:, i], x) + mxnet.nd.dot(logxcountneg[:, i], 1-x)
    # normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpx -= mxnet.nd.max(logpx)
    # and compute the softmax using logpx
    px = mxnet.nd.exp(logpx).asnumpy()
    px /= numpy.sum(px)

    # bar chart and image of digit
    figarr[1, ctr].bar(range(10), px)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1
    if ctr == 10:
        break
matplotlib.pyplot.show()

for i in range(10):
    print(random.random())

for i in range(10):
    print(random.randint(1, 100))

counts = numpy.zeros(100)
fig, axes = matplotlib.pyplot.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# mangle subplots such that we can index them in a linear fashion rather than
# a 2d grid
for i in range(1, 1000001):
    counts[random.randint(0, 99)] += 1
    if i in [10, 100, 1000, 10000, 100000, 1000000]:
        axes[int(math.log10(i))-1].bar(numpy.arange(1, 101), counts)
matplotlib.pyplot.show()

# number of samples
n = 1000000
y = numpy.random.uniform(0, 1, n)
x = numpy.arange(1, n+1)
# count number of occurrences and divide by the number of total draws
p0 = numpy.cumsum(y < 0.35) / x
p1 = numpy.cumsum(y >= 0.35) / x
matplotlib.pyplot.figure(figsize=(15, 8))
matplotlib.pyplot.semilogx(x, p0)
matplotlib.pyplot.semilogx(x, p1)
matplotlib.pyplot.show()

x = numpy.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * numpy.exp(-0.5 * x**2)
matplotlib.pyplot.figure(figsize=(10, 5))
matplotlib.pyplot.plot(x, p)
matplotlib.pyplot.show()

# generate 10 random sequences of 10,000 random normal variables N(0,1)
tmp = numpy.random.uniform(size=(10000,10))
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8)
mean = 1 * 0.5 + 2 * 0.2
variance = 1 * 0.5 + 4 * 0.2 - mean**2
print('mean {}, variance {}'.format(mean, variance))
# cumulative sum and normalization
y = numpy.arange(1,10001).reshape(10000,1)
z = numpy.cumsum(x,axis=0) / y
matplotlib.pyplot.figure(figsize=(10,5))
for i in range(10):
    matplotlib.pyplot.semilogx(y,z[:,i])
matplotlib.pyplot.semilogx(y,(variance**0.5) * numpy.power(y,-0.5) + mean,'r')
matplotlib.pyplot.semilogx(y,-(variance**0.5) * numpy.power(y,-0.5) + mean,'r')
matplotlib.pyplot.show()