import tensorflow_model_optimization
import numpy
import tensorflow
import zipfile
import os

tensorflow.enable_eager_execution()

batch_size = 128
epochs = 12

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

end_step = numpy.ceil(1.0 * x_train.shape[0] / batch_size).astype(numpy.int32) * epochs

pruning_params = {'pruning_schedule': tensorflow_model_optimization.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.50,
    final_sparsity=0.90,
    begin_step=2000,
    end_step=end_step,
    frequency=100)}

pruned_model = tensorflow.keras.Sequential([
    tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(tensorflow.keras.layers.Conv2D(32, 5, padding='same', activation='relu'), input_shape=(28, 28, 1), **pruning_params),
    tensorflow.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(tensorflow.keras.layers.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
    tensorflow.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    tensorflow.keras.layers.Flatten(),
    tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(tensorflow.keras.layers.Dense(1024, activation='relu'), **pruning_params),
    tensorflow.keras.layers.Dropout(0.4),
    tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(tensorflow.keras.layers.Dense(10, activation='softmax'), **pruning_params)])

pruned_model.summary()

pruned_model.compile(
    loss=tensorflow.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    tensorflow_model_optimization.sparsity.keras.UpdatePruningStep(),
    tensorflow_model_optimization.sparsity.keras.PruningSummaries(log_dir='sparse2', profile_batch=0)]

pruned_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_test, y_test))

score = pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# saved_model() sets include_optimizer to True by default. Spelling it out here to highlight.
tensorflow.keras.models.save_model(pruned_model, 'sparse2/1.h5', include_optimizer=True)

with tensorflow_model_optimization.sparsity.keras.prune_scope():
    restored_model = tensorflow.keras.models.load_model('sparse2/1.h5')

restored_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=2,
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_test, y_test))

score = restored_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

final_model = tensorflow_model_optimization.sparsity.keras.strip_pruning(pruned_model)
final_model.summary()

# No need to save the optimizer with the graph for serving.
tensorflow.keras.models.save_model(final_model, 'sparse2/2.h5', include_optimizer=False)

with zipfile.ZipFile('sparse1/1.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write('sparse1/1.h5')
print("Size of the unpruned model before compression: %.2f Mb" % (os.path.getsize('sparse1/1.h5') / float(2**20)))
print("Size of the unpruned model after compression: %.2f Mb" % (os.path.getsize('sparse1/1.zip') / float(2**20)))

with zipfile.ZipFile('sparse2/2.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write('sparse2/2.h5')
print("Size of the pruned model before compression: %.2f Mb" % (os.path.getsize('sparse2/2.h5') / float(2**20)))
print("Size of the pruned model after compression: %.2f Mb" % (os.path.getsize('sparse2/2.zip') / float(2 ** 20)))