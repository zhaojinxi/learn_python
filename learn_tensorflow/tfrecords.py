import os
import tensorflow
import PIL

cwd = os.getcwd()

#
writer = tensorflow.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = PIL.Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tensorflow.train.Example(features=tensorflow.train.Features(feature={"label": tensorflow.train.Feature(int64_list=tensorflow.train.Int64List(value=[index])), 'img_raw': tensorflow.train.Feature(bytes_list=tensorflow.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())
writer.close()

#
for serialized_example in tensorflow.python_io.tf_record_iterator("train.tfrecords"):
    example = tensorflow.train.Example()
    example.ParseFromString(serialized_example)
    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    print(image, label)

#
def read_and_decode(filename):
    filename_queue = tensorflow.train.string_input_producer([filename])
    reader = tensorflow.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tensorflow.parse_single_example(serialized_example, features={'label': tensorflow.FixedLenFeature([], tensorflow.int64), 'img_raw' : tensorflow.FixedLenFeature([], tensorflow.string),})
    img = tensorflow.decode_raw(features['img_raw'], tensorflow.uint8)
    img = tensorflow.reshape(img, [224, 224, 3])
    img = tensorflow.cast(img, tensorflow.float32) * (1. / 255) - 0.5
    label = tensorflow.cast(features['label'], tensorflow.int32)
    return img, label

#
img, label = read_and_decode("train.tfrecords")
img_batch, label_batch = tensorflow.train.shuffle_batch([img, label], batch_size=30, capacity=2000, min_after_dequeue=1000)
init = tensorflow.initialize_all_variables()
with tensorflow.Session() as sess:
    sess.run(init)
    threads = tensorflow.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])
        print(val.shape, l)