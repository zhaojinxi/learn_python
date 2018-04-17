import tensorflow
import numpy
import dataset
import os
import time

tensorflow.enable_eager_execution()

tensorflow.executing_eagerly()
x = [[2.]]
m = tensorflow.matmul(x, x)
print("hello, {}".format(m))

a = tensorflow.constant([[1, 2], [3, 4]])
print(a)
# Broadcasting support
b = tensorflow.add(a, 1)
print(b)
# Operator overloading is supported
print(a * b)
# Use NumPy values
c = numpy.multiply(a, b)
print(c)
# Obtain numpy value from a tensor:
print(a.numpy())

w = tensorflow.contrib.eager.Variable([[1.0]])
with tensorflow.contrib.eager.GradientTape() as tape:
  loss = w * w
grad = tape.gradient(loss, [w])
print(grad)

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tensorflow.random_normal([NUM_EXAMPLES])
noise = tensorflow.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise
def prediction(input, weight, bias):
  return input * weight + bias
# A loss function using mean-squared error
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tensorflow.reduce_mean(tensorflow.square(error))
# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tensorflow.contrib.eager.GradientTape() as tape:
    loss_value = loss(weights, biases) 
  return tape.gradient(loss_value, [weights, biases])
train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tensorflow.contrib.eager.Variable(5.)
B = tensorflow.contrib.eager.Variable(10.)
print("Initial loss: {:.3f}".format(loss(W, B)))
for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))
print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))

# dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
# data.train.labels))
# ...
# for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
#   ...
#   with tfe.GradientTape() as tape:
#     logits = model(images, training=True)
#     loss_value = loss(logits, labels)
#   ...
#   grads = tape.gradient(loss_value, model.variables)
#   optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())

def line_search_step(fn, init_x, rate=1.0):
  with tensorflow.contrib.eager.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad, = tape.gradient(value, [init_x])
  grad_norm = tensorflow.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value

def square(x):
  return tensorflow.multiply(x, x)
grad = tensorflow.contrib.eager.gradients_function(square)
square(3.)  # => 9.0
grad(3.)    # => [6.0]
# The second-order derivative of square:
gradgrad = tensorflow.contrib.eager.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]
# The third-order derivative is None:
gradgradgrad = tensorflow.contrib.eager.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]
# With flow control:
def abs(x):
  return x if x > 0. else -x
grad = tensorflow.contrib.eager.gradients_function(abs)
grad(3.)   # => [1.0]
grad(-3.)  # => [-1.0]

@tensorflow.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tensorflow.identity(x)
  def grad_fn(dresult):
    return [tensorflow.clip_by_norm(dresult, norm), None]
  return y, grad_fn

def log1pexp(x):
  return tensorflow.log(1 + tensorflow.exp(x))
grad_log1pexp = tensorflow.contrib.eager.gradients_function(log1pexp)
# The gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]
# However, x = 100 fails because of numerical instability.
grad_log1pexp(100.)  # => [nan]

@tensorflow.custom_gradient
def log1pexp(x):
  e = tensorflow.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tensorflow.log(1 + e), grad
grad_log1pexp = tensorflow.contrib.eager.gradients_function(log1pexp)
# As before, the gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]
# And the gradient computation also works at x = 100.
grad_log1pexp(100.)  # => [1.0]

model = tensorflow.keras.Sequential([
  tensorflow.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tensorflow.keras.layers.Dense(10)])

class MNISTModel(tensorflow.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tensorflow.keras.layers.Dense(units=10)
    self.dense2 = tensorflow.keras.layers.Dense(units=10)
  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result
model = MNISTModel()

# Create a tensor representing a blank image
batch = tensorflow.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)
result = model(batch)
# => tensorflow.Tensor([[[ 0.  0., ..., 0.]]], shape=(1, 1, 10), dtype=float32)

dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)

def loss(model, x, y):
  prediction = model(x)
  return tensorflow.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)
def grad(model, inputs, targets):
  with tensorflow.contrib.eager.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.001)
x, y = tensorflow.contrib.eager.Iterator(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))
# Training loop
for (i, (x, y)) in enumerate(tensorflow.contrib.eager.Iterator(dataset_train)):
  # Calculate derivatives of the input function with respect to its parameters.
  grads = grad(model, x, y)
  # Apply the gradient to the model
  optimizer.apply_gradients(zip(grads, model.variables), global_step=tensorflow.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))
print("Final loss: {:.3f}".format(loss(model, x, y)))

# with tensorflow.device("/gpu:0"):
#   for (i, (x, y)) in enumerate(tensorflow.contrib.eager.Iterator(dataset_train)):
#     # minimize() is equivalent to the grad() and apply_gradients() calls.
#     optimizer.minimize(lambda: loss(model, x, y), global_step=tensorflow.train.get_or_create_global_step())

class Model(tensorflow.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tensorflow.contrib.eager.Variable(5., name='weight')
    self.B = tensorflow.contrib.eager.Variable(10., name='bias')
  def predict(self, inputs):
    return inputs * self.W + self.B
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tensorflow.random_normal([NUM_EXAMPLES])
noise = tensorflow.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise
# The loss function to be optimized
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tensorflow.reduce_mean(tensorflow.square(error))
def grad(model, inputs, targets):
  with tensorflow.contrib.eager.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])
# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01)
print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]), global_step=tensorflow.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))
print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

with tensorflow.device("gpu:0"):
  v = tensorflow.contrib.eager.Variable(tensorflow.random_normal([1000, 1000]))
  v = None  # v no longer takes up GPU memory

x = tensorflow.contrib.eager.Variable(10.)
checkpoint = tensorflow.contrib.eager.Checkpoint(x=x)  # save as "x"
x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')
x.assign(11.)  # Change the variable after saving.
# Restore values from the checkpoint
checkpoint.restore(save_path)
print(x)  # => 2.0  

model = Model()
optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = './'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tensorflow.contrib.eager.Checkpoint(optimizer=optimizer, model=model, optimizer_step=tensorflow.train.get_or_create_global_step())
root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))

m = tensorflow.contrib.eager.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5

# writer = tensorflow.contrib.summary.create_file_writer(logdir)
# global_step=tensorflow.train.get_or_create_global_step()  # return global step var
# writer.set_as_default()
# for _ in range(iterations):
#   global_step.assign_add(1)
#   # Must include a record_summaries method
#   with tensorflow.contrib.summary.record_summaries_every_n_global_steps(100):
#     # your model code goes here
#     tensorflow.contrib.summary.scalar('loss', loss)

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tensorflow.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tensorflow.matmul(x, x)
    _ = x.numpy()  # Make sure to execute op and not just enqueue it
  end = time.time()
  return end - start
shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))
# Run on CPU:
with tensorflow.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tensorflow.random_normal(shape), steps)))
# Run on GPU, if available:
if tensorflow.contrib.eager.num_gpus() > 0:
  with tensorflow.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tensorflow.random_normal(shape), steps)))
else:
  print("GPU: not found")

x = tensorflow.random_normal([10, 10])
x_gpu0 = x.gpu()
x_cpu = x.cpu()
_ = tensorflow.matmul(x_cpu, x_cpu)    # Runs on CPU
_ = tensorflow.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0
if tensorflow.contrib.eager.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tensorflow.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1

def my_py_func(x):
  x = tensorflow.matmul(x, x)  # You can use tf ops
  print(x)  # but it's eager!
  return x
with tensorflow.Session() as sess:
  x = tensorflow.placeholder(dtype=tensorflow.float32)
  # Call eager function in graph!
  pf = tensorflow.contrib.eager.py_func(my_py_func, [x], tensorflow.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]