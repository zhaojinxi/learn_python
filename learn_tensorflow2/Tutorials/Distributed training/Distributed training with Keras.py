import os
import tensorflow
import tensorflow_datasets

tensorflow_datasets.disable_progress_bar()

print(tensorflow.__version__)

datasets, info = tensorflow_datasets.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

strategy = tensorflow.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# You can also do info.splits.total_num_examples to get the total number of examples in the dataset.

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def scale(image, label):
    image = tensorflow.cast(image, tensorflow.float32)
    image /= 255

    return image, label

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

with strategy.scope():
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tensorflow.keras.layers.MaxPooling2D(),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(10)])

    model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

callbacks = [
        tensorflow.keras.callbacks.TensorBoard(log_dir='./logs'),
        tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
        tensorflow.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()]

model.fit(train_dataset, epochs=12, callbacks=callbacks)

model.load_weights(tensorflow.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

path = 'saved_model/'

model.save(path, save_format='tf')

unreplicated_model = tensorflow.keras.models.load_model(path)

unreplicated_model.compile( loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

with strategy.scope():
    replicated_model = tensorflow.keras.models.load_model(path)
    replicated_model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print ('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))