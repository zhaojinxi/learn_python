import tensorflow
import numpy
import os

'''

'''

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
log_dir = 'log/'
model_dir = 'model/'
batch_size = 128
repeat = 10
init_lr = 0.01
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

logits = cnn(input_image)

test_predict = tensorflow.argmax(tensorflow.nn.softmax(logits), 1)

Saver = tensorflow.train.Saver()
Session = tensorflow.Session()
Saver.restore(Session, tensorflow.train.latest_checkpoint(model_dir))

g = tensorflow.get_default_graph()
##########################
tensorflow.contrib.quantize.create_eval_graph(input_graph=g)
# tensorflow.train.write_graph(g.as_graph_def(), model_dir, 'eval_graph_def.pb')
with open('model/eval_graph_def.pb', 'w') as f:
    f.write(str(g.as_graph_def()))
##############################

os.system('''
    freeze_graph \
        --input_graph=model/eval_graph_def.pb \
        --input_checkpoint=model/cnn-4680 \
        --output_graph=model/frozen_eval_graph.pb \
        --output_node_names=ArgMax
    ''')

# os.system('''
#     toco \
#         --graph_def_file=model/frozen_eval_graph.pb \
#         --output_file=model/converted_model.tflite \
#         --input_format=TENSORFLOW_GRAPHDEF \
#         --output_format=TFLITE \
#         --inference_type=QUANTIZED_UINT8 \
#         --input_shape="1,28,28,1" \
#         --input_array=input_image \
#         --output_array=ArgMax \
#         --std_dev_values=1 \
#         --mean_value=0
#     ''')

converter = tensorflow.lite.TFLiteConverter.from_frozen_graph("model/frozen_eval_graph.pb", ["input_image"], ["ArgMax"])
tflite_model = converter.convert()
open("model/converted_model.tflite", "wb").write(tflite_model)