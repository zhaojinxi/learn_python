import tensorflow
import numpy
import os
import quant_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_dir = 'model/'

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()
train_image = train_image.reshape(-1, 28, 28, 1)
test_image = test_image.reshape(-1, 28, 28, 1)
# train_image = (train_image - 128) / 128
# test_image = (test_image - 128) / 128
train_label = train_label.astype(numpy.int32)
test_label = test_label.astype(numpy.int32)

input_image = tensorflow.placeholder(tensorflow.float32, [None, 28, 28, 1], name='input_image')

logits = quant_model.cnn(input_image)

test_predict = tensorflow.argmax(tensorflow.nn.softmax(logits), 1)

Saver = tensorflow.train.Saver()
Session = tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())
Saver.restore(Session, tensorflow.train.latest_checkpoint(model_dir))
# print(Session.run(tensorflow.get_default_graph().get_tensor_by_name('cnn/conv1/b1:0')))

total = 0
for image, label in zip(test_image, test_label):
    output_data = Session.run(test_predict, feed_dict={input_image: image.reshape(1, 28, 28, 1)})
    if output_data[0] == label:
        total = total + 1
print(total / test_image.shape[0])