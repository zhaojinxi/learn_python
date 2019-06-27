import numpy
import tensorflow
import os

tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_dir = 'estimator_model/'
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

def train_input_fn():
    train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(total_data).repeat(repeat).batch(batch_size).prefetch(batch_size)
    return train_dataset

def eval_input_fn():
    test_dataset = tensorflow.data.Dataset.from_tensor_slices((test_image, test_label)).batch(batch_size).prefetch(batch_size)
    return test_dataset

def cnn_model_fn(features, labels, mode):
    conv1 = tensorflow.layers.conv2d(
        inputs=features,
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
    dropout = tensorflow.layers.dropout(inputs=dense, rate=0.4, training=mode == tensorflow.estimator.ModeKeys.TRAIN)

    logits = tensorflow.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tensorflow.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tensorflow.nn.softmax(logits, name="softmax_tensor")}

    if mode == tensorflow.estimator.ModeKeys.PREDICT:
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tensorflow.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": tensorflow.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

mnist_classifier = tensorflow.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

logging_hook = tensorflow.estimator.LoggingTensorHook(tensors={"probabilities": "softmax_tensor"}, every_n_iter=1000)

mnist_classifier.train(input_fn=lambda: train_input_fn(), hooks=[logging_hook])

eval_results = mnist_classifier.evaluate(input_fn=lambda: eval_input_fn())
print(eval_results)