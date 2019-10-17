import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
    datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    return train_dataset, eval_dataset

def get_model():
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')])

        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        return model

model = get_model()
train_dataset, eval_dataset = get_data()
model.fit(train_dataset, epochs=2)

keras_model_path = "keras_save/"
model.save(keras_model_path)  # save() should be called out of strategy scope

restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs=2)

another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
with another_strategy.scope():
    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
    restored_keras_model_ds.fit(train_dataset, epochs=2)

model = get_model()  # get a fresh model
saved_model_path = "tf_save/"
tf.saved_model.save(model, saved_model_path)

DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

predict_dataset = eval_dataset.map(lambda image, label: image)
for batch in predict_dataset.take(1):
    print(inference_func(batch))

another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

    dist_predict_dataset = another_strategy.experimental_distribute_dataset(predict_dataset)

    # Calling the function in a distributed manner
    for batch in dist_predict_dataset:
        another_strategy.experimental_run_v2(inference_func, args=(batch,))

import tensorflow_hub as hub

def build_model(loaded):
    x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
    # Wrap what's loaded to a KerasLayer
    keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
    model = tf.keras.Model(x, keras_layer)
    return model

another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    model = build_model(loaded)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(train_dataset, epochs=2)

model = get_model()

# Saving the model using Keras's save() API
model.save(keras_model_path)

another_strategy = tf.distribute.MirroredStrategy()
# Loading the model using lower level API
with another_strategy.scope():
    loaded = tf.saved_model.load(keras_model_path)

class SubclassedModel(tf.keras.Model):
    output_name = 'output_layer'
    def __init__(self):
        super(SubclassedModel, self).__init__()
        self._dense_layer = tf.keras.layers.Dense(5, dtype=tf.dtypes.float32, name=self.output_name)

    def call(self, inputs):
        return self._dense_layer(inputs)

my_model = SubclassedModel()
# my_model.save(keras_model_path)  # ERROR!
tf.saved_model.save(my_model, saved_model_path)