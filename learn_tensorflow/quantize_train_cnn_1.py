import tensorflow
import numpy
import os
import quantize_train_cnn_0

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
# train_image = (train_image - 128) / 128
# test_image = (test_image - 128) / 128
train_label = train_label.astype(numpy.int32)
test_label = test_label.astype(numpy.int32)

input_image = tensorflow.placeholder(tensorflow.float32, [None, 28, 28, 1], name='input_image')
input_label = tensorflow.placeholder(tensorflow.int32, [None], name='input_label')
training = tensorflow.placeholder(tensorflow.bool, name='training')
global_step = tensorflow.train.get_or_create_global_step()
learning_rate = tensorflow.train.exponential_decay(init_lr, global_step, max_step, decay_rate)

logits = quantize_train_cnn_0.cnn(input_image, training)

test_predict = tensorflow.argmax(tensorflow.nn.softmax(logits), 1)

loss = tensorflow.losses.sparse_softmax_cross_entropy(input_label, logits)

g = tensorflow.get_default_graph()
tensorflow.contrib.quantize.create_training_graph(input_graph=g, quant_delay=int(total_data / batch_size * repeat * 0.8))

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    # minimize = tensorflow.contrib.opt.NadamOptimizer(learning_rate).minimize(loss, global_step=global_step, name='minimize')
    minimize = tensorflow.train.GradientDescentOptimizer(learning_rate=init_lr).minimize(loss, global_step=global_step, name='minimize')

test_accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.cast(test_predict, tensorflow.int32), input_label), tensorflow.float32))

tensorflow.summary.scalar('test_accuracy', test_accuracy)
merge_all = tensorflow.summary.merge_all()

Saver = tensorflow.train.Saver(max_to_keep=1)

Session = tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

FileWriter = tensorflow.summary.FileWriter(model_dir, Session.graph)

num = total_data // batch_size
for i in range(max_step + 1):
    if Session.run(global_step) % int(total_data / batch_size) == 0:
        summary = Session.run(merge_all, feed_dict={input_image: test_image, input_label: test_label, training: False})
        FileWriter.add_summary(summary, Session.run(global_step))
        Saver.save(Session, model_dir + 'cnn', global_step)

    temp_image = train_image[i % num * batch_size : i % num * batch_size + batch_size, :]
    temp_label = train_label[i % num * batch_size : i % num * batch_size + batch_size]
    Session.run(minimize, feed_dict={input_image: temp_image, input_label: temp_label, training: True})
    print(Session.run(global_step))