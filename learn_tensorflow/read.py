import argparse
import os
import sys
import time
import tensorflow
from tensorflow.examples.tutorials.mnist import mnist

with tensorflow.Graph().as_default():
    with tensorflow.name_scope('train_data'):
        filename_queue = tensorflow.train.string_input_producer([os.path.join('/tmp/data', 'train.tfrecords')], num_epochs=1)
        reader = tensorflow.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tensorflow.parse_single_example(serialized_example, features={'image_raw': tensorflow.FixedLenFeature([], tensorflow.string), 'label': tensorflow.FixedLenFeature([], tensorflow.int64),})
        image = tensorflow.decode_raw(features['image_raw'], tensorflow.uint8)
        image.set_shape([784])
        image = tensorflow.cast(image, tensorflow.float32) * (1. / 255) - 0.5
        label = tensorflow.cast(features['label'], tensorflow.int32)
        train_images, train_labels = tensorflow.train.shuffle_batch([image, label], batch_size=100, num_threads=2, capacity=1000 + 3 * 100, min_after_dequeue=1000)

    logits = tensorflow.examples.tutorials.mnist.mnist.inference(train_images, 128, 32)
    loss = tensorflow.examples.tutorials.mnist.mnist.loss(logits, train_labels)
    train_op = tensorflow.examples.tutorials.mnist.mnist.training(loss, 0.01)
    init_op = tensorflow.group(tensorflow.global_variables_initializer(), tensorflow.local_variables_initializer())
    with tensorflow.Session() as sess:
        sess.run(init_op)
        coord = tensorflow.train.Coordinator()
        threads = tensorflow.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                step += 1
        except tensorflow.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (1, step))
        finally:
            coord.request_stop()
        coord.join(threads)