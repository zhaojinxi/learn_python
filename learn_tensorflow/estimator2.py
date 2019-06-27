import tensorflow
import pandas as pd

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

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tensorflow.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
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

# Fetch the data
(train_x, train_y), (test_x, test_y) = load_data()

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tensorflow.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# Train the Model.
classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, batch_size), steps=train_steps)

# Evaluate the model.
eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1]}

predictions = classifier.predict(input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=batch_size))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('\nPrediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))