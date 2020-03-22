import os
import numpy
import tensorflow

(train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()
train_images = numpy.expand_dims(train_images, -1) / 255.0
test_images = numpy.expand_dims(test_images, -1) / 255.0

batch_size = 1000
epoch = 100
step = train_images.shape[0] // batch_size
learning_rate = 0.001 / 256 * batch_size

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Conv2D(64, 3, 2),
    tensorflow.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
    tensorflow.keras.layers.ReLU(),
    tensorflow.keras.layers.Conv2D(128, 3, 2),
    tensorflow.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
    tensorflow.keras.layers.ReLU(),
    tensorflow.keras.layers.Conv2D(256, 3, 2),
    tensorflow.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
    tensorflow.keras.layers.ReLU(),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(10)])

lr = tensorflow.keras.experimental.CosineDecay(learning_rate, step * epoch)
model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(lr),
    loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy()])

for i in range(epoch):
    model.fit(train_images, train_labels, batch_size)
    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size)
    model.reset_metrics()

tensorflow.keras.models.save_model(model, 'tensorflow2_mnist', overwrite=True, include_optimizer=False)