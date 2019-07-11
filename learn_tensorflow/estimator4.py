import tensorflow
import numpy

tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()
train_image = train_image.reshape(-1, 28, 28).astype(numpy.float32)
train_image = (train_image - 127.5) / 128

mnist_ds = tensorflow.data.Dataset.from_tensor_slices((train_image)).batch(1)

def representative_data_gen():
  for input_value in mnist_ds.take(60000):
    yield [input_value]

converter = tensorflow.lite.TFLiteConverter.from_saved_model('estimator_model/1562834322')
#设置是否量化权重
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
#设置是否量化激活
# converter.representative_dataset = representative_data_gen
#设置是否仅支持整数量化
# converter.target_spec = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
open("estimator_model/converted_model.tflite", "wb").write(tflite_model)