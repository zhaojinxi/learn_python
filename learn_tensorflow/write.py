"""Converts MNIST data to TFRecords file format with Example protos."""

import argparse
import os
import sys
import tensorflow
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

FLAGS = None

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tensorflow.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    label=tensorflow.train.Feature(int64_list=tensorflow.train.Int64List(value=[int(labels[index])]))
    image_raw=tensorflow.train.Feature(bytes_list=tensorflow.train.BytesList(value=[image_raw]))
    features=tensorflow.train.Features(feature={'label':label,'image_raw':image_raw})
    example = tensorflow.train.Example(features=features)
    writer.write(example.SerializeToString())
  writer.close()

def main(unused_argv):
  # Get the data.
  data_sets = read_data_sets(FLAGS.directory, dtype=tensorflow.uint8, reshape=False, validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--directory', type=str, default='/tmp/data', help='Directory to download data files and write the converted result')
  parser.add_argument('--validation_size', type=int, default=5000, help="""Number of examples to separate from the training data for the validation set.""")
  FLAGS, unparsed = parser.parse_known_args()
  tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)