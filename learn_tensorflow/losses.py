import tensorflow
import numpy

numpy.random.seed(0)
a1 = numpy.random.randint(0, 2, [3, 2, 3])
a2 = numpy.random.randint(0, 2, [3, 2, 3])
w1 = numpy.random.choice([0, 1], [3, 2, 3])
w2 = numpy.random.choice([0, 1], [3, 2])
b1 = numpy.array([[[0, 1], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [0, 1]]])
b2 = numpy.random.rand(3, 2, 2)

labels = tensorflow.placeholder(tensorflow.float32, None, 'labels')
predictions = tensorflow.placeholder(tensorflow.float32, None, 'predictions')
weights = tensorflow.placeholder(tensorflow.float32, None, 'weights')

# MEAN, NONE, SUM, SUM_BY_NONZERO_WEIGHTS, SUM_OVER_BATCH_SIZE, SUM_OVER_NONZERO_WEIGHTS
mean_squared_error = tensorflow.losses.mean_squared_error(labels, predictions, weights, reduction=tensorflow.losses.Reduction.SUM_OVER_BATCH_SIZE)
absolute_difference = tensorflow.losses.absolute_difference(labels, predictions, weights, reduction=tensorflow.losses.Reduction.SUM_OVER_BATCH_SIZE)
softmax_cross_entropy = tensorflow.losses.softmax_cross_entropy(labels, predictions, weights, reduction=tensorflow.losses.Reduction.MEAN)

Session = tensorflow.Session()

for lab, pre, wei in zip(a1, a2, w1):
    print('labels:',lab)
    print('predictions:',pre)
    print('weights:',wei)
    for one in [mean_squared_error, absolute_difference]:
        y = Session.run(one, feed_dict={labels: lab, predictions: pre, weights: wei})
        print('%s' % one.name, y)