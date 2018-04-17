import tensorflow
import numpy

tensorflow.enable_eager_execution()

z_dimensions = 100
batch_size = 50

(train_x, train_y), (test_x, test_y) =tensorflow.keras.datasets.mnist.load_data("mnist.npz")

with tensorflow.name_scope('discriminator'):
    d_w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([5, 5, 1, 32], stddev=0.1),name='d_w1')
    d_b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[32]),name='d_b1')
    d_w2 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([5, 5, 32, 64], stddev=0.1),name='d_w2')
    d_b2 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[64]),name='d_b2')
    d_w3 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),name='d_w3')
    d_b3 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[1024]),name='d_b3')
    d_w4 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([1024, 2], stddev=0.1),name='d_w4')
    d_b4 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[2]),name='d_b4')

with tensorflow.name_scope('generator'):
    g_w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([z_dimensions, 3136], stddev=0.1),name='g_w1')
    g_b1 = tensorflow.Variable(tensorflow.constant(0.1,shape=[3136]),name='g_b1')
    g_w2 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([3, 3, 1, z_dimensions//2], stddev=0.1),name='g_w2')
    g_b2 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[z_dimensions//2]),name='g_b2')
    g_w3 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([3, 3, z_dimensions//2, z_dimensions//4], stddev=0.1),name='g_w3')
    g_b3 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[z_dimensions//4]),name='g_b3')
    g_w4 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([1, 1, z_dimensions//4, 1], stddev=0.1),name='g_w4')
    g_b4 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1,shape=[1]),name='g_b4')

def discriminator(x):
    d1 = tensorflow.nn.conv2d(input=X, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = tensorflow.nn.relu(d1 + d_b1)
    d1 = tensorflow.nn.max_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    d2 = tensorflow.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = tensorflow.nn.relu(d2 + d_b2)
    d2 = tensorflow.nn.max_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    d2 = tensorflow.reshape(d2, [-1, 7 * 7 * 64])

    d3 = tensorflow.matmul(d2, d_w3)
    d3 = tensorflow.nn.relu(d3 + d_b3)

    d4 = tensorflow.matmul(d3, d_w4) + d_b4

    return d4

def generator(z):
    g1 = tensorflow.matmul(z, g_w1) + g_b1
    g1 = tensorflow.nn.relu(g1)
    g1 = tensorflow.reshape(g1, [-1, 56, 56, 1])

    g2 = tensorflow.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = tensorflow.nn.relu(g2 + g_b2)

    g3 = tensorflow.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = tensorflow.nn.relu(g3 + g_b3)

    g4 = tensorflow.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = tensorflow.sigmoid(g4 + g_b4)

Gz = generator(z_placeholder)
Dx = discriminator(x_placeholder)
Dg = discriminator(Gz)
d_loss_real = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tensorflow.ones_like(Dx)))
d_loss_fake = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tensorflow.zeros_like(Dg)))
g_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tensorflow.ones_like(Dg)))

num = x_train.shape[0] // batch_size

for i in range(300):
    z_batch = numpy.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = train_x[i % num * batch_size:i % num * batch_size + batch_size,:].reshape(-1,28,28,1)