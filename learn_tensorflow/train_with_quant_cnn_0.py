import tensorflow

def prelu(x, name):
    alpha = tensorflow.get_variable("alpha%s" % name, x.get_shape()[-1], tensorflow.float32, tensorflow.constant_initializer(0.25))
    result = tensorflow.nn.relu(x) - alpha * tensorflow.nn.relu(-x)
    return result

def hard_swish(x):
    y = x * (tensorflow.nn.relu6(x + 3)) / 6
    return y

# def cnn(x, training):
#     with tensorflow.variable_scope('cnn', reuse=tensorflow.AUTO_REUSE):
#         with tensorflow.variable_scope('conv1'):
#             w1 = tensorflow.get_variable('w1', [3, 3, 1, 8], initializer=tensorflow.initializers.lecun_normal())
#             b1 = tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))
#             z1 = tensorflow.nn.conv2d(x, w1, [1, 2, 2, 1], 'SAME') + b1

#             z1 = tensorflow.layers.batch_normalization(z1, training=training)

#             z1 = tensorflow.nn.relu(z1)
#             # z1 = hard_swish(z1)
#             # z1 = prelu(z1, 1)

#             # wd1 = tensorflow.get_variable('wd1', [3, 3, 8, 1], initializer=tensorflow.initializers.lecun_normal())
#             # bd1 = tensorflow.get_variable('bd1', 8, initializer=tensorflow.initializers.zeros)
#             # rd1 = tensorflow.nn.depthwise_conv2d(z1, wd1, [1, 1, 1, 1], 'SAME') + bd1
#             # z1 = tensorflow.nn.relu(rd1)

#         with tensorflow.variable_scope('conv2'):
#             w2 = tensorflow.get_variable('w2', [3, 3, 8, 16], initializer=tensorflow.initializers.lecun_normal())
#             b2 = tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
#             z2 = tensorflow.nn.conv2d(z1, w2, [1, 2, 2, 1], 'SAME') + b2
#             z2 = tensorflow.nn.relu(z2)

#         with tensorflow.variable_scope('conv3'):
#             w3 = tensorflow.get_variable('w3', [3, 3, 16, 32], initializer=tensorflow.initializers.lecun_normal())
#             b3 = tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))
#             z3 = tensorflow.nn.conv2d(z2, w3, [1, 2, 2, 1], 'VALID') + b3
#             z3 = tensorflow.nn.relu(z3)

#         with tensorflow.variable_scope('conv4'):
#             w4 = tensorflow.get_variable('w4', [3, 3, 32, 10], initializer=tensorflow.initializers.lecun_normal())
#             b4 = tensorflow.get_variable('b4', 10, initializer=tensorflow.constant_initializer(0))
#             z4 = tensorflow.nn.conv2d(z3, w4, [1, 1, 1, 1], 'VALID') + b4
#             z4 = tensorflow.nn.relu(z4)

#         with tensorflow.variable_scope('output'):
#             z4 = tensorflow.reshape(z4, [-1, 10])
#     return z4

def cnn(x, training):
    conv1 = tensorflow.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tensorflow.nn.relu)
    pool1 = tensorflow.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tensorflow.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tensorflow.nn.relu)
    pool2 = tensorflow.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tensorflow.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tensorflow.layers.dense(inputs=pool2_flat, units=1024, activation=tensorflow.nn.relu)
    dropout = tensorflow.layers.dropout(inputs=dense, rate=0.4, training=training)

    return dropout

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
    data = tensorflow.random.uniform([4, 28, 28, 1])
    predict = cnn(data, True)