import tensorflow
import numpy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
log_dir = 'log/'
model_dir = 'model/'
batch_size = 128
repeat = 10
init_lr = 0.001
decay_rate = 0.1
total_data = 60000
max_step = numpy.ceil(total_data * repeat / batch_size).astype(numpy.int32)

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()
train_image = train_image.reshape(-1, 28, 28, 1).astype(numpy.float32)
test_image = test_image.reshape(-1, 28, 28, 1).astype(numpy.float32)
train_label = train_label.astype(numpy.int32)
test_label = test_label.astype(numpy.int32)

def cnn(x):
    with tensorflow.variable_scope('cnn', reuse=tensorflow.AUTO_REUSE):
        with tensorflow.variable_scope('conv1'):
            w1 = tensorflow.get_variable('w1', [3, 3, 1, 8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b1 = tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))
            z1 = tensorflow.nn.conv2d(x, w1, [1, 2, 2, 1], 'SAME') + b1
            z1 = tensorflow.nn.leaky_relu(z1)

        with tensorflow.variable_scope('conv2'):
            w2 = tensorflow.get_variable('w2', [3, 3, 8, 16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b2 = tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
            z2 = tensorflow.nn.conv2d(z1, w2, [1, 2, 2, 1], 'SAME') + b2
            z2 = tensorflow.nn.leaky_relu(z2)

        with tensorflow.variable_scope('conv3'):
            w3 = tensorflow.get_variable('w3', [3, 3, 16, 32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b3 = tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))
            z3 = tensorflow.nn.conv2d(z2, w3, [1, 2, 2, 1], 'VALID') + b3
            z3 = tensorflow.nn.leaky_relu(z3)

        with tensorflow.variable_scope('conv4'):
            w4 = tensorflow.get_variable('w4', [3, 3, 32, 10], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b4 = tensorflow.get_variable('b4', 10, initializer=tensorflow.constant_initializer(0))
            z4 = tensorflow.nn.conv2d(z3, w4, [1, 1, 1, 1], 'VALID') + b4
            z4 = tensorflow.nn.leaky_relu(z4)

        with tensorflow.variable_scope('output'):
            z4 = tensorflow.reshape(z4, [-1, 10])
    return z4

input_image = tensorflow.placeholder(tensorflow.float32, [None, 28, 28, 1], name='input_image')
input_label = tensorflow.placeholder(tensorflow.int32, [None], name='input_label')
global_step = tensorflow.train.get_or_create_global_step()
learning_rate = tensorflow.train.exponential_decay(init_lr, global_step, max_step, decay_rate)

logits = cnn(input_image)

test_predict = tensorflow.argmax(tensorflow.nn.softmax(logits), 1)

loss = tensorflow.losses.sparse_softmax_cross_entropy(input_label, logits)

g = tensorflow.get_default_graph()
##########################
tensorflow.contrib.quantize.create_training_graph(input_graph=g, quant_delay=int(total_data / batch_size * repeat / 2))
##############################

minimize = tensorflow.contrib.opt.NadamOptimizer(learning_rate).minimize(loss, global_step=global_step, name='minimize')

test_accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.cast(test_predict, tensorflow.int32), input_label), tensorflow.float32))

tensorflow.summary.scalar('test_accuracy', test_accuracy)
merge_all = tensorflow.summary.merge_all()

Saver = tensorflow.train.Saver(max_to_keep=1)

Session = tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

num = total_data // batch_size
for i in range(max_step):
    if Session.run(global_step) % int(total_data / batch_size) == 0:
        summary = Session.run(merge_all, feed_dict={input_image: test_image, input_label: test_label})
        FileWriter.add_summary(summary, Session.run(global_step))
        Saver.save(Session, model_dir+'cnn', global_step)

    temp_image = train_image[i % num * batch_size : i % num * batch_size + batch_size, :]
    temp_label = train_label[i % num * batch_size : i % num * batch_size + batch_size]
    Session.run(minimize, feed_dict={input_image: temp_image, input_label: temp_label})
    print(Session.run(global_step))