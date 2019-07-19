import tensorflow_model_optimization
import numpy
import tensorflow
import zipfile
import os

tensorflow.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

converter = tensorflow.lite.TFLiteConverter.from_keras_model_file('sparse3/1.h5')

converter.optimizations = [tensorflow.lite.Optimize.OPTIMIZE_FOR_LATENCY]

tflite_quant_model = converter.convert()

with open('sparse5/1.tflite', 'wb') as f:
    f.write(tflite_quant_model)

with zipfile.ZipFile('sparse5/1.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write('sparse5/1.tflite')
print("Size of the tflite model before compression: %.2f Mb" % (os.path.getsize('sparse5/1.tflite') / float(2**20)))
print("Size of the tflite model after compression: %.2f Mb" % (os.path.getsize('sparse5/1.zip') / float(2 ** 20)))

interpreter = tensorflow.lite.Interpreter(model_path=str('sparse5/1.tflite'))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def eval_model(interpreter, x_test, y_test):
    total_seen = 0
    num_correct = 0

    for img, label in zip(x_test, y_test):
        inp = img.reshape((1, 28, 28, 1))
        total_seen += 1
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        if numpy.argmax(predictions) == numpy.argmax(label):
            num_correct += 1

        if total_seen % 1000 == 0:
            print("Accuracy after %i images: %f" % (total_seen, float(num_correct) / float(total_seen)))
    return float(num_correct) / float(total_seen)

print(eval_model(interpreter, x_test, y_test))