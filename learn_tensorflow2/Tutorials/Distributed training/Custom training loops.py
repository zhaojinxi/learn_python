import os
import numpy
import tensorflow

print(tensorflow.__version__)

fashion_mnist = tensorflow.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / numpy.float32(255)
test_images = test_images / numpy.float32(255)

# If the list of devices is not specified in the `tensorflow.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tensorflow.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tensorflow.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

def create_model():
    model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
            tensorflow.keras.layers.MaxPooling2D(),
            tensorflow.keras.layers.Conv2D(64, 3, activation='relu'),
            tensorflow.keras.layers.MaxPooling2D(),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(64, activation='relu'),
            tensorflow.keras.layers.Dense(10)])
    return model

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and divide by global batch size.
    loss_object = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tensorflow.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tensorflow.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    test_loss = tensorflow.keras.metrics.Mean(name='test_loss')
    train_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
    model = create_model()
    optimizer = tensorflow.keras.optimizers.Adam()
    checkpoint = tensorflow.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
    def train_step(inputs):
        images, labels = inputs

        with tensorflow.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tensorflow.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
        return strategy.reduce(tensorflow.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tensorflow.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}")
        print (template.format(epoch+1, train_loss, train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

eval_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

new_model = create_model()
new_optimizer = tensorflow.keras.optimizers.Adam()

test_dataset = tensorflow.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

@tensorflow.function
def eval_step(images, labels):
    predictions = new_model(images, training=False)
    eval_accuracy(labels, predictions)

checkpoint = tensorflow.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
    eval_step(images, labels)

print ('Accuracy after restoring the saved model without strategy: {}'.format(eval_accuracy.result() * 100))

with strategy.scope():
    for _ in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        train_iter = iter(train_dist_dataset)

        for _ in range(10):
            total_loss += distributed_train_step(next(train_iter))
            num_batches += 1
        average_train_loss = total_loss / num_batches

        template = ("Epoch {}, Loss: {}, Accuracy: {}")
        print (template.format(epoch+1, average_train_loss, train_accuracy.result()*100))
        train_accuracy.reset_states()

with strategy.scope():
    @tensorflow.function
    def distributed_train_epoch(dataset):
        total_loss = 0.0
        num_batches = 0
        for x in dataset:
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(x,))
            total_loss += strategy.reduce(tensorflow.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            num_batches += 1
        return total_loss / tensorflow.cast(num_batches, dtype=tensorflow.float32)

    for epoch in range(EPOCHS):
        train_loss = distributed_train_epoch(train_dist_dataset)

        template = ("Epoch {}, Loss: {}, Accuracy: {}")
        print (template.format(epoch+1, train_loss, train_accuracy.result()*100))

        train_accuracy.reset_states()