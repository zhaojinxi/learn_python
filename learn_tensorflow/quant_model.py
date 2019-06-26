import tensorflow

def cnn(x):
    with tensorflow.variable_scope('cnn', reuse=tensorflow.AUTO_REUSE):
        with tensorflow.variable_scope('conv1'):
            w1 = tensorflow.get_variable('w1', [3, 3, 1, 8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b1 = tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))
            z1 = tensorflow.nn.conv2d(x, w1, [1, 2, 2, 1], 'SAME') + b1
            z1 = tensorflow.nn.relu6(z1)

        with tensorflow.variable_scope('conv2'):
            w2 = tensorflow.get_variable('w2', [3, 3, 8, 16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b2 = tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
            z2 = tensorflow.nn.conv2d(z1, w2, [1, 2, 2, 1], 'SAME') + b2
            z2 = tensorflow.nn.relu6(z2)

        with tensorflow.variable_scope('conv3'):
            w3 = tensorflow.get_variable('w3', [3, 3, 16, 32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b3 = tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))
            z3 = tensorflow.nn.conv2d(z2, w3, [1, 2, 2, 1], 'VALID') + b3
            z3 = tensorflow.nn.relu6(z3)

        with tensorflow.variable_scope('conv4'):
            w4 = tensorflow.get_variable('w4', [3, 3, 32, 10], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b4 = tensorflow.get_variable('b4', 10, initializer=tensorflow.constant_initializer(0))
            z4 = tensorflow.nn.conv2d(z3, w4, [1, 1, 1, 1], 'VALID') + b4
            z4 = tensorflow.nn.relu6(z4)

        with tensorflow.variable_scope('output'):
            z4 = tensorflow.reshape(z4, [-1, 10])
    return z4