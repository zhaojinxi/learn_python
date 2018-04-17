import mxnet

mxnet.random.seed(1)

x = mxnet.nd.array([[1, 2], [3, 4]])

x.attach_grad()

with mxnet.autograd.record():
    y = x * 2
    z = y * x

z.backward()

print(x.grad)

with mxnet.autograd.record():
    y = x * 2
    z = y * x
head_gradient = mxnet.nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print(x.grad)

a = mxnet.nd.random_normal(shape=3)
a.attach_grad()
with mxnet.autograd.record():
    b = a * 2
    while (mxnet.nd.norm(b) < 1000).asscalar():
        b = b * 2

    if (mxnet.nd.sum(b) > 0).asscalar():
        c = b
    else:
        c = 100 * b

head_gradient = mxnet.nd.array([0.01, 1.0, .1])
c.backward(head_gradient)

print(a.grad)