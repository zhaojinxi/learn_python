import tensorflow
import pandas

tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)

SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

batch_size = 100
train_steps = 1000

def load_data(y_name='Species'):
    TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
    TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
    train_path = tensorflow.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tensorflow.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    train = pandas.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pandas.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat(10).batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tensorflow.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

def _parse_line(line):
    # Decode the line into its fields
    CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
    fields = tensorflow.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tensorflow.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def my_model(features, labels, mode, params):
    net = tensorflow.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tensorflow.layers.dense(net, units=units, activation=tensorflow.nn.relu)

    logits = tensorflow.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tensorflow.argmax(logits, 1)
    if mode == tensorflow.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tensorflow.newaxis],
            'probabilities': tensorflow.nn.softmax(logits),
            'logits': logits,}
        return tensorflow.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tensorflow.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tensorflow.summary.scalar('accuracy', accuracy[1])

    if mode == tensorflow.estimator.ModeKeys.EVAL:
        return tensorflow.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # assert mode == tensorflow.estimator.ModeKeys.TRAIN
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

(train_x, train_y), (test_x, test_y) = load_data()

my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))

classifier = tensorflow.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[10, 10], n_classes=3)
# classifier = tensorflow.estimator.Estimator(model_fn=my_model, params={'feature_columns': my_feature_columns, 'hidden_units': [10, 10], 'n_classes': 3,})

classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=train_steps)

eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1]}

predictions = classifier.predict(input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=batch_size))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('\nPrediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))