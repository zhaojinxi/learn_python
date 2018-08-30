import tensorflow as tf
import numpy as np
import time

tf.enable_eager_execution()
tfe = tf.contrib.eager

tf.executing_eagerly()        # => True
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"

a = tf.constant([[1, 2],[3, 4]])
print(a)
# Broadcasting support
b = tf.add(a, 1)
print(b)
# Operator overloading is supported
print(a * b)
# Use NumPy values
c = np.multiply(a, b)
print(c)
# Obtain numpy value from a tensor:
print(a.numpy())

def fizzbuzz(max_num):
  counter = tf.constant(0)
  for num in range(max_num):
    num = tf.constant(num)
    if num % 3 == 0 and num % 5 == 0:
      print('FizzBuzz')
    elif num % 3 == 0:
      print('Fizz')
    elif num % 5 == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter

class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    self.output_units = output_units
  def build(self, input):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence remove the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
      "kernel", [input.shape[-1], self.output_units])
  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)])

class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)
  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result
model = MNISTModel()

w = tfe.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w
grad = tape.gradient(loss, [w])
print(grad)

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise
def prediction(input, weight, bias):
  return input * weight + bias
# A loss function using mean-squared error
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))
# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tf.GradientTape() as tape:
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])
train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tfe.Variable(5.)
B = tfe.Variable(10.)
print("Initial loss: {:.3f}".format(loss(W, B)))
for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))
print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))

# dataset = tf.data.Dataset.from_tensor_slices((data.train.images,data.train.labels))
# ...
# for (batch, (images, labels)) in enumerate(dataset):
#   ...
#   with tf.GradientTape() as tape:
#     logits = model(images, training=True)
#     loss_value = loss(logits, labels)
#   ...
#   grads = tape.gradient(loss_value, model.variables)
#   optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())

# Create a tensor representing a blank image
batch = tf.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)
result = model(batch)

import dataset  # download dataset.py file
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)

def loss(model, x, y):
  prediction = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
x, y = iter(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))
# Training loop
for (i, (x, y)) in enumerate(dataset_train):
  # Calculate derivatives of the input function with respect to its parameters.
  grads = grad(model, x, y)
  # Apply the gradient to the model
  optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))
print("Final loss: {:.3f}".format(loss(model, x, y)))

with tf.device("/gpu:0"):
  for (i, (x, y)) in enumerate(dataset_train):
    # minimize() is equivalent to the grad() and apply_gradients() calls.
    optimizer.minimize(lambda: loss(model, x, y),global_step=tf.train.get_or_create_global_step())

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')
  def predict(self, inputs):
    return inputs * self.W + self.B
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise
# The loss function to be optimized
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tf.reduce_mean(tf.square(error))
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])
# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))
print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

with tf.device("gpu:0"):
  v = tfe.Variable(tf.random_normal([1000, 1000]))
  v = None  # v no longer takes up GPU memory

x = tfe.Variable(10.)
checkpoint = tfe.Checkpoint(x=x)  # save as "x"
x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')
x.assign(11.)  # Change the variable after saving.
# Restore values from the checkpoint
checkpoint.restore(save_path)
print(x)  # => 2.0

model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = '/path/to/model_dir'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer, model=model,optimizer_step=tf.train.get_or_create_global_step())
root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5

# writer = tf.contrib.summary.create_file_writer(logdir)
# global_step=tf.train.get_or_create_global_step()  # return global step var
# writer.set_as_default()
# for _ in range(iterations):
#   global_step.assign_add(1)
#   # Must include a record_summaries method
#   with tf.contrib.summary.record_summaries_every_n_global_steps(100):
#     # your model code goes here
#     tf.contrib.summary.scalar('loss', loss)
#      ...

def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad, = tape.gradient(value, [init_x])
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value

def square(x):
  return tf.multiply(x, x)
grad = tfe.gradients_function(square)
square(3.)  # => 9.0
grad(3.)    # => [6.0]
# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]
# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]
# With flow control:
def abs(x):
  return x if x > 0. else -x
grad = tfe.gradients_function(abs)
grad(3.)   # => [1.0]
grad(-3.)  # => [-1.0]

@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn

def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)
# The gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]
# However, x = 100 fails because of numerical instability.
grad_log1pexp(100.)  # => [nan]

@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad
grad_log1pexp = tfe.gradients_function(log1pexp)
# As before, the gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]
# And the gradient computation also works at x = 100.
grad_log1pexp(100.)  # => [1.0]

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
    _ = x.numpy()  # Make sure to execute op and not just enqueue it
  end = time.time()
  return end - start
shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))
# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))
# Run on GPU, if available:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
  print("GPU: not found")

x = tf.random_normal([10, 10])
x_gpu0 = x.gpu()
x_cpu = x.cpu()
_ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0
if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1

def my_py_func(x):
  x = tf.matmul(x, x)  # You can use tf ops
  print(x)  # but it's eager!
  return x
with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # Call eager function in graph!
  pf = tfe.py_func(my_py_func, [x], tf.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]  