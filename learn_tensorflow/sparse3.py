import tensorflow_model_optimization
import numpy
import tensorflow
import zipfile
import os

tensorflow.enable_eager_execution()

batch_size = 128
epochs = 4

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

# Load the serialized model
loaded_model = tensorflow.keras.models.load_model('sparse1/1.h5')

end_step = numpy.ceil(1.0 * x_train.shape[0] / batch_size).astype(numpy.int32) * epochs

new_pruning_params = {'pruning_schedule': tensorflow_model_optimization.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.50, final_sparsity=0.90,
    begin_step=0,
    end_step=end_step,
    frequency=100)}

new_pruned_model = tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(loaded_model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss=tensorflow.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    tensorflow_model_optimization.sparsity.keras.UpdatePruningStep(),
    tensorflow_model_optimization.sparsity.keras.PruningSummaries(log_dir='sparse3', profile_batch=0)]

new_pruned_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_test, y_test))

score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

final_model = tensorflow_model_optimization.sparsity.keras.strip_pruning(new_pruned_model)
final_model.summary()

tensorflow.keras.models.save_model(final_model, 'sparse3/1.h5', include_optimizer=False)

with zipfile.ZipFile('sparse3/1.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write('sparse3/1.h5')
print("Size of the pruned model before compression: %.2f Mb" % (os.path.getsize('sparse3/1.h5') / float(2**20)))
print("Size of the pruned model after compression: %.2f Mb" % (os.path.getsize('sparse3/1.zip') / float(2 ** 20)))