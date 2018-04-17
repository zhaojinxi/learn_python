import mxnet

##########################
# Instantiate two scalars
##########################
x = mxnet.nd.array([3.0])
y = mxnet.nd.array([2.0])
##########################
# Add them
##########################
print('x + y = ', x + y)
##########################
# Multiply them
##########################
print('x * y = ', x * y)
##########################
# Divide x by y
##########################
print('x / y = ', x / y)
##########################
# Raise x to the power y.
##########################
print('x ** y = ', mxnet.nd.power(x,y))

x.asscalar()

u = mxnet.nd.arange(4)
print('u = ', u)

u[3]

len(u)

u.shape

a = 2
x = mxnet.nd.array([1,2,3])
y = mxnet.nd.array([10,20,30])
print(a * x)
print(a * x + y)

A = mxnet.nd.zeros((5,4))
A

x = mxnet.nd.arange(20)
A = x.reshape((5, 4))
A

print('A[2, 3] = ', A[2, 3])

print('row 2', A[2, :])
print('column 3', A[:, 3])

A.T

X = mxnet.nd.arange(24).reshape((2, 3, 4))
print('X.shape =', X.shape)
print('X =', X)

u = mxnet.nd.array([1, 2, 4, 8])
v = mxnet.nd.ones_like(u) * 2
print('v =', v)
print('u + v', u + v)
print('u - v', u - v)
print('u * v', u * v)
print('u / v', u / v)

B = mxnet.nd.ones_like(A) * 3
print('B =', B)
print('A + B =', A + B)
print('A * B =', A * B)

a = 2
x = mxnet.nd.ones(3)
y = mxnet.nd.zeros(3)
print(x.shape)
print(y.shape)
print((a * x).shape)
print((a * x + y).shape)

mxnet.nd.sum(u)

mxnet.nd.sum(A)

print(mxnet.nd.mean(A))
print(mxnet.nd.sum(A) / A.size)

mxnet.nd.dot(u, v)

mxnet.nd.sum(u * v)

mxnet.nd.dot(A, u)

A = mxnet.nd.ones(shape=(3, 4))
B = mxnet.nd.ones(shape=(4, 5))
mxnet.nd.dot(A, B)

mxnet.nd.norm(u)

mxnet.nd.sum(mxnet.nd.abs(u))