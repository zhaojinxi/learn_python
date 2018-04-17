import mxnet as mx
from mxnet import nd
mx.random.seed(1)

x = nd.empty((3, 4))
print(x)

x = nd.zeros((3, 5))
print(x)

x = nd.ones((3, 4))
print(x)

y = nd.random_normal(0, 1, shape=(3, 4))
print(y)

print(y.shape)

print(y.size)

x + y

x * y

nd.exp(y)

nd.dot(x, y.T)

print('id(y):', id(y))
y = y + x
print('id(y):', id(y))

print('id(y):', id(y))
y[:] = x + y
print('id(y):', id(y))

nd.elemwise_add(x, y, out=y)

print('id(x):', id(x))
x += y
x
print('id(x):', id(x))

x[1:3]

x[1,2] = 9.0
x

x[1:2,1:3]

x[1:2,1:3] = 5.0
x

x = nd.ones(shape=(3,3))
print('x = ', x)
y = nd.arange(3)
print('y = ', y)
print('x + y = ', x + y)

y = y.reshape((3,1))
print('y = ', y)
print('x + y = ', x+y)

a = x.asnumpy()
type(a)

y = nd.array(a)
y

z = nd.ones(shape=(3,3), ctx=mx.gpu(0))
z


x_gpu = x.copyto(mx.gpu(0))
print(x_gpu)

x_gpu + z

print(x_gpu.context)
print(z.context)

print('id(z):', id(z))
z = z.copyto(mx.gpu(0))
print('id(z):', id(z))
z = z.as_in_context(mx.gpu(0))
print('id(z):', id(z))
print(z)