import tensorflow

a1 = [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
a2 = [[1, 1, 0, 0, 0], [1, 1, 0, 1, 1]]
a3 = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
a4 = [[1, 1, 1, 1, 1], [0, 0, 1, 1, 0]]
a5 = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1]]
a = [a1, a2, a3, a4, a5]

labels = tensorflow.placeholder(tensorflow.float32, [5], 'labels')
predictions = tensorflow.placeholder(tensorflow.float32, [5], 'predictions')

true_positives = tensorflow.metrics.true_positives(labels, predictions, name='true_positives')
true_negatives = tensorflow.metrics.true_negatives(labels, predictions, name='true_negatives')
false_positives = tensorflow.metrics.false_positives(labels, predictions, name='false_positives')
false_negatives = tensorflow.metrics.false_negatives(labels, predictions, name='false_negatives')
accuracy = tensorflow.metrics.accuracy(labels, predictions, name='accuracy')
precision = tensorflow.metrics.precision(labels, predictions, name='precision')
recall = tensorflow.metrics.recall(labels, predictions, name='recall')
mean_absolute_error = tensorflow.metrics.mean_absolute_error(labels, predictions, name='mean_absolute_error')
mean_squared_error = tensorflow.metrics.mean_squared_error(labels, predictions, name='mean_squared_error')
root_mean_squared_error = tensorflow.metrics.root_mean_squared_error(labels, predictions, name='root_mean_squared_error')

Session = tensorflow.Session()

for epoch in range(3):
    Session.run(tensorflow.variables_initializer(tensorflow.get_collection(tensorflow.GraphKeys.METRIC_VARIABLES)))
    for one_data in a:
        for one_metric in [accuracy, precision, recall, mean_absolute_error, mean_squared_error]:
            _, y = Session.run(one_metric, feed_dict={labels: one_data[0], predictions: one_data[1]})
            print('%s' % one_metric[1].name, y)