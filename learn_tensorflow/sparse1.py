import tensorflow

tensorflow.enable_eager_execution()

batch_size = 128
epochs = 10

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
    tensorflow.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
    tensorflow.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(1024, activation='relu'),
    tensorflow.keras.layers.Dropout(0.4),
    tensorflow.keras.layers.Dense(10, activation='softmax')])

model.summary()

callbacks = [tensorflow.keras.callbacks.TensorBoard(log_dir='sparse1', profile_batch=0)]

model.compile(
    loss=tensorflow.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Backend agnostic way to save/restore models
tensorflow.keras.models.save_model(model, 'sparse1/1.h5', include_optimizer=False)