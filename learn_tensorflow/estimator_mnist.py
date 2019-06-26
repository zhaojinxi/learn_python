import tensorflow
import numpy

log_train_dir = 'log_train/'
log_test_dir = 'log_test/'
model_dir = 'model/'
batch_size = 64
repeat = 5
max_step = 60000 / batch_size * repeat
lr = 0.01
decay_rate = 0.1

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()
train_image = train_image.reshape(-1, 28, 28, 1).astype(numpy.float32)
test_image = test_image.reshape(-1, 28, 28, 1).astype(numpy.float32)
train_label = train_label.astype(numpy.int32)
test_label = test_label.astype(numpy.int32)

def cnn(features, labels, mode, params):
    x = tensorflow.keras.Input(shape=(28, 28, 1))

    y = tensorflow.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        strides=2,
        padding='valid')(x)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=2,
        padding='valid')(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='valid')(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.Conv2D(
        filters=10,
        kernel_size=2,
        strides=2,
        padding='valid')(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.Flatten()(y)

    model = tensorflow.keras.Model(inputs=x, outputs=y)
    return model

def model_fn(image, labels, mode, params):
    model = cnn()

    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=lr)

        logits = model(image, training=True)
        loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tensorflow.metrics.accuracy(labels=labels, predictions=tensorflow.argmax(logits, axis=1))

        tensorflow.identity(lr, 'learning_rate')
        tensorflow.identity(loss, 'cross_entropy')
        tensorflow.identity(accuracy[1], name='train_accuracy')

        tensorflow.summary.scalar('train_accuracy', accuracy[1])

        return tensorflow.estimator.EstimatorSpec(
            mode=tensorflow.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tensorflow.train.get_or_create_global_step()))

    if mode == tensorflow.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tensorflow.estimator.EstimatorSpec(
            mode=tensorflow.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': tensorflow.metrics.accuracy(labels=labels, predictions=tensorflow.argmax(logits, axis=1)),})

    if mode == tensorflow.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tensorflow.argmax(logits, axis=1),
            'probabilities': tensorflow.nn.softmax(logits),}
        return tensorflow.estimator.EstimatorSpec(
            mode=tensorflow.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={'classify': tensorflow.estimator.export.PredictOutput(predictions)})

session_config = tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True))

def get_distribution_strategy(distribution_strategy="default",
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None,
                              num_packs=1):
    """Return a DistributionStrategy for running the model.

    Args:
        distribution_strategy: a string specifying which distribution strategy to use. Accepted values are 'off', 'default', 'one_device', 'mirrored', 'parameter_server', 'multi_worker_mirrored', case insensitive. 'off' means not to use Distribution Strategy; 'default' means to choose from `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy` according to the number of GPUs and number of workers.
        num_gpus: Number of GPUs to run this model.
        num_workers: Number of workers to run this model.
        all_reduce_alg: Optional. Specifies which algorithm to use when performing all-reduce. For `MirroredStrategy`, valid values are "nccl" and "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are "ring" and "nccl". If None, DistributionStrategy will choose based on device topology.
        num_packs: Optional. Sets the `num_packs` in `tensorflow.distribute.NcclAllReduce` or `tensorflow.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.

    Returns:
        tensorflow.distribute.DistibutionStrategy object.
    Raises:
        ValueError: if `distribution_strategy` is 'off' or 'one_device' and
        `num_gpus` is larger than 1; or `num_gpus` is negative.
    """
    if num_gpus < 0:
        raise ValueError("`num_gpus` can not be negative.")

    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == "off":
        if num_gpus > 1 or num_workers > 1:
            raise ValueError("When {} GPUs and  {} workers are specified, distribution_strategy flag cannot be set to 'off'.".format(num_gpus, num_workers))
        return None

    if distribution_strategy == "multi_worker_mirrored" or num_workers > 1:
        return tensorflow.distribute.experimental.MultiWorkerMirroredStrategy(
            communication=_collective_communication(all_reduce_alg))

    if (distribution_strategy == "one_device" or
        (distribution_strategy == "default" and num_gpus <= 1)):
        if num_gpus == 0:
            return tensorflow.distribute.OneDeviceStrategy("device:CPU:0")
        else:
            if num_gpus > 1:
                raise ValueError("`OneDeviceStrategy` can not be used for more than "
                            "one device.")
            return tensorflow.distribute.OneDeviceStrategy("device:GPU:0")

    if distribution_strategy in ("mirrored", "default"):
        if num_gpus == 0:
            assert distribution_strategy == "mirrored"
            devices = ["device:CPU:0"]
        else:
            devices = ["device:GPU:%d" % i for i in range(num_gpus)]
        return tensorflow.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

    if distribution_strategy == "parameter_server":
        return tensorflow.distribute.experimental.ParameterServerStrategy()

    raise ValueError("Unrecognized Distribution Strategy: %r" % distribution_strategy)

distribution_strategy = get_distribution_strategy(
    distribution_strategy='default',
    num_gpus='1',
    all_reduce_alg=None)

run_config = tensorflow.estimator.RunConfig(train_distribute=distribution_strategy, session_config=session_config)

classifier = tensorflow.estimator.Estimator(
    model_fn=model_function,
    model_dir=model_dir,
    config=run_config,
    params={'data_format': 'channels_first'})

def train_input_fn():
    train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(60000).batch(batch_size)
    return train_dataset

def eval_input_fn():
    test_dataset = tensorflow.data.Dataset.from_tensor_slices((test_image, test_label)).batch(batch_size)
    return test_dataset

def get_train_hooks(name_list, use_tpu=False, **kwargs):
    """Factory for getting a list of TensorFlow hooks for training by name.

    Args:
        name_list: a list of strings to name desired hook classes. Allowed: LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook, which are defined as keys in HOOKS
        use_tpu: Boolean of whether computation occurs on a TPU. This will disable hooks altogether.
        **kwargs: a dictionary of arguments to the hooks.

    Returns:
        list of instantiated hooks, ready to be used in a classifier.train call.

    Raises:
        ValueError: if an unrecognized name is passed.
    """

    if not name_list:
        return []

    if use_tpu:
        tensorflow.compat.v1.logging.warning('hooks_helper received name_list `{}`, but a TPU is specified. No hooks will be used.'.format(name_list))
        return []

    train_hooks = []
    for name in name_list:
        hook_name = HOOKS.get(name.strip().lower())
        if hook_name is None:
            raise ValueError('Unrecognized training hook requested: {}'.format(name))
        else:
            train_hooks.append(hook_name(**kwargs))

    return train_hooks

def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None, **kwargs):  # pylint: disable=unused-argument
    """Function to get LoggingTensorHook.

    Args:
        every_n_iter: `int`, print the values of `tensors` once every N local
        steps taken on the current worker.
        tensors_to_log: List of tensor names or dictionary mapping labels to tensor
        names. If not set, log _TENSORS_TO_LOG by default.
        **kwargs: a dictionary of arguments to LoggingTensorHook.

    Returns:
        Returns a LoggingTensorHook with a standard set of tensors that will be
        printed to stdout.
    """
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG

    return tensorflow.estimator.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=every_n_iter)

def get_profiler_hook(model_dir, save_steps=1000, **kwargs):  # pylint: disable=unused-argument
    """Function to get ProfilerHook.

    Args:
        model_dir: The directory to save the profile traces to.
        save_steps: `int`, print profile traces every N steps.
        **kwargs: a dictionary of arguments to ProfilerHook.

    Returns:
        Returns a ProfilerHook that writes out timelines that can be loaded into
        profiling tools like chrome://tracing.
    """
    return tensorflow.estimator.ProfilerHook(save_steps=save_steps, output_dir=model_dir)

def get_examples_per_second_hook(every_n_steps=100,
                                 batch_size=128,
                                 warm_steps=5,
                                 **kwargs):  # pylint: disable=unused-argument
    """Function to get ExamplesPerSecondHook.

    Args:
        every_n_steps: `int`, print current and average examples per second every
        N steps.
        batch_size: `int`, total batch size used to calculate examples/second from
        global time.
        warm_steps: skip this number of steps before logging and running average.
        **kwargs: a dictionary of arguments to ExamplesPerSecondHook.

    Returns:
        Returns a ProfilerHook that writes out timelines that can be loaded into
        profiling tools like chrome://tracing.
    """
    return hooks.ExamplesPerSecondHook(
        batch_size=batch_size, every_n_steps=every_n_steps,
        warm_steps=warm_steps, metric_logger=logger.get_benchmark_logger())

def get_logging_metric_hook(tensors_to_log=None,
                            every_n_secs=600,
                            **kwargs):  # pylint: disable=unused-argument
    """Function to get LoggingMetricHook.

    Args:
        tensors_to_log: List of tensor names or dictionary mapping labels to tensor names. If not set, log _TENSORS_TO_LOG by default.
        every_n_secs: `int`, the frequency for logging the metric. Default to every 10 mins.
        **kwargs: a dictionary of arguments.

    Returns:
        Returns a LoggingMetricHook that saves tensor values in a JSON format.
    """
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG
    return metric_hook.LoggingMetricHook(
        tensors=tensors_to_log,
        metric_logger=logger.get_benchmark_logger(),
        every_n_secs=every_n_secs)

HOOKS = {
    'loggingtensorhook': get_logging_tensor_hook,
    'profilerhook': get_profiler_hook,
    'examplespersecondhook': get_examples_per_second_hook,
    'loggingmetrichook': get_logging_metric_hook}

train_hooks = get_train_hooks(
      'loggingtensorhook',
      model_dir=model_dir,
      batch_size=batch_size)

for _ in range(repeat):
    classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print('\nEvaluation results:\n\t%s\n' % eval_results)

image = tensorflow.placeholder(tensorflow.float32, [None, 28, 28])
input_fn = tensorflow.estimator.export.build_raw_serving_input_receiver_fn({'image': image})
mnist_classifier.export_savedmodel(model_dir, input_fn, strip_default_attrs=True)