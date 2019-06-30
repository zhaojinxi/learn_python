import tensorflow
import numpy
import os
import train_with_quant_cnn_0

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_dir = 'model/'

input_image = tensorflow.placeholder(tensorflow.float32, [None, 28, 28, 1], name='input_image')
input_label = tensorflow.placeholder(tensorflow.int32, [None], name='input_label')

logits = train_with_quant_cnn_0.cnn(input_image, False)

test_predict = tensorflow.argmax(tensorflow.nn.softmax(logits), 1)

Saver = tensorflow.train.Saver()
Session = tensorflow.Session()
Saver.restore(Session, tensorflow.train.latest_checkpoint(model_dir))

g = tensorflow.get_default_graph()
tensorflow.contrib.quantize.create_eval_graph(input_graph=g)
tensorflow.train.write_graph(g.as_graph_def(), model_dir, 'eval_graph_def.pb')

os.system('''
    freeze_graph \
        --input_graph=model/eval_graph_def.pb \
        --input_checkpoint=model/cnn-4680 \
        --output_graph=model/frozen_eval_graph.pb \
        --output_node_names=ArgMax
    ''')

os.system('''
    toco \
        --graph_def_file=model/frozen_eval_graph.pb \
        --output_file=model/converted_model.tflite \
        --input_format=TENSORFLOW_GRAPHDEF \
        --output_format=TFLITE \
        --inference_type=QUANTIZED_UINT8 \
        --input_shape="1,28,28,1" \
        --input_array=input_image \
        --output_array=ArgMax \
        --std_dev_values=1 \
        --mean_value=0
    ''')