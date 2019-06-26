import tensorflow
import numpy
import tqdm
import os
import sys
import pandas

tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name='converted_model'
model_dir = 'model/'

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()
train_image = train_image.reshape(-1, 28, 28, 1)
test_image = test_image.reshape(-1, 28, 28, 1)
train_label = train_label.astype(numpy.int32)
test_label = test_label.astype(numpy.int32)

interpreter = tensorflow.lite.Interpreter(model_path=os.path.join(model_dir, model_name) + '.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

for image, label in zip(test_image, test_label):
    interpreter.set_tensor(input_index, image.reshape(1, 28, 28, 1))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)