import tensorflow
import sklearn.preprocessing
import numpy

tensorflow.enable_eager_execution()

log_train_dir = 'log_train/'
log_test_dir = 'log_test/'
model_dir = 'model/cnn'
batch_size = 64
epoch = 3
max_step = 60000 / batch_size * epoch
init_lr = 0.001
decay_rate = 0.1

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()
train_image = train_image.reshape(-1, 28, 28, 1).astype(numpy.float32)
test_image = test_image.reshape(-1, 28, 28, 1).astype(numpy.float32)
train_label = train_label.astype(numpy.int32)
test_label = test_label.astype(numpy.int32)

train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(60000).batch(batch_size)
test_dataset = tensorflow.data.Dataset.from_tensor_slices((test_image, test_label)).batch(batch_size)

class CNN(tensorflow.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        with tensorflow.variable_scope('cnn'):
            self.w1 = tensorflow.get_variable('w1', [3, 3, 1, 8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            self.b1 = tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))

            self.w2 = tensorflow.get_variable('w2', [3, 3, 8, 16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            self.b2 = tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))

            self.w3 = tensorflow.get_variable('w3', [3, 3, 16, 32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            self.b3 = tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))

            self.w4 = tensorflow.get_variable('w4', [3, 3, 32, 10], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            self.b4 = tensorflow.get_variable('b4', 10, initializer=tensorflow.constant_initializer(0))

    def call(self, x, training):
        with tensorflow.variable_scope('cnn'):
            z1 = tensorflow.nn.conv2d(x, self.w1, [1, 2, 2, 1], 'SAME') + self.b1
            z1 = tensorflow.nn.selu(z1)

            z2 = tensorflow.nn.conv2d(z1, self.w2, [1, 2, 2, 1], 'SAME') + self.b2
            z2 = tensorflow.nn.selu(z2)

            z3 = tensorflow.nn.conv2d(z2, self.w3, [1, 2, 2, 1], 'VALID') + self.b3
            z3 = tensorflow.nn.selu(z3)

            z4 = tensorflow.nn.conv2d(z3, self.w4, [1, 1, 1, 1], 'VALID') + self.b4
            z4 = tensorflow.nn.selu(z4)

            z4 = tensorflow.reshape(z4, [-1, 10])
            if training == False:
                z4 = tensorflow.nn.softmax(z4)
        return z4

model = CNN()

global_step = tensorflow.train.get_or_create_global_step()
learning_rate = tensorflow.train.exponential_decay(init_lr, global_step, max_step, decay_rate)

Optimizer = tensorflow.train.AdamOptimizer(learning_rate)

summary_writer = tensorflow.contrib.summary.create_file_writer(log_train_dir, flush_millis=10000)
test_summary_writer = tensorflow.contrib.summary.create_file_writer(log_test_dir, flush_millis=10000)

checkpoint = tensorflow.train.Checkpoint(model=model, optimizer=Optimizer, step_counter=global_step)
checkpoint.restore(tensorflow.train.latest_checkpoint(model_dir))

def compute_accuracy(logits, labels):
    predictions = tensorflow.argmax(logits, axis=1, output_type=tensorflow.int64)
    labels = tensorflow.cast(labels, tensorflow.int64)
    return tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predictions, labels), dtype=tensorflow.float32))

def loss(logits, labels):
    return tensorflow.reduce_mean(tensorflow.losses.sparse_softmax_cross_entropy(labels,logits))

def train(model, optimizer, dataset, global_step):
    for (batch, (images, labels)) in enumerate(dataset):
        with tensorflow.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(logits, labels)
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))

        with summary_writer.as_default():
            with tensorflow.contrib.summary.record_summaries_every_n_global_steps(100, global_step=global_step):
                if tensorflow.contrib.summary.should_record_summaries():
                    tensorflow.contrib.summary.scalar('loss', loss_value)
                    tensorflow.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))

def test(model, dataset):
    avg_loss = tensorflow.contrib.eager.metrics.Mean('loss', dtype=tensorflow.float32)
    accuracy = tensorflow.contrib.eager.metrics.Accuracy('accuracy', dtype=tensorflow.float32)

    for (images, labels) in dataset:
        logits = model(images, training=False)
        avg_loss(loss(logits, labels))
        accuracy(tensorflow.argmax(logits, axis=1, output_type=tensorflow.int64), tensorflow.cast(labels, tensorflow.int64))
    print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' % (avg_loss.result(), 100 * accuracy.result()))

    with test_summary_writer.as_default():
        with tensorflow.contrib.summary.always_record_summaries():
            tensorflow.contrib.summary.scalar('loss', avg_loss.result())
            tensorflow.contrib.summary.scalar('accuracy', accuracy.result())

for _ in range(epoch):
    train(model, Optimizer, train_dataset, global_step)
    print('\nTrain time for epoch #%d (%d total steps)' % (checkpoint.save_counter.numpy() + 1, global_step.numpy()))

    test(model, test_dataset)
    checkpoint.save(model_dir)