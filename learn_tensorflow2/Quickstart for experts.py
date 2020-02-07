import tensorflow

mnist = tensorflow.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tensorflow.newaxis]
x_test = x_test[..., tensorflow.newaxis]

train_ds = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(tensorflow.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tensorflow.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tensorflow.keras.layers.Flatten()
        self.d1 = tensorflow.keras.layers.Dense(128, activation='relu')
        self.d2 = tensorflow.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tensorflow.keras.losses.SparseCategoricalCrossentropy()

optimizer = tensorflow.keras.optimizers.Adam()

train_loss = tensorflow.keras.metrics.Mean(name='train_loss')
train_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tensorflow.keras.metrics.Mean(name='test_loss')
test_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tensorflow.function
def train_step(images, labels):
    with tensorflow.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tensorflow.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))