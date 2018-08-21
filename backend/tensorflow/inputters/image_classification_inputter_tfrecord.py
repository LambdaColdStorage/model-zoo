"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF inputter interfaces for image classification (tfrecord real data)
"""
from __future__ import print_function
import os
import csv

import tensorflow as tf

from backend.tensorflow.inputters import inputter


class Inputter(inputter.Inputter):
  def __init__(self, config):
    super(Inputter, self).__init__(config)

  def input_fn(self, mode, test_samples=[]):
    """Implementation of input_fn
    Returns:
      A data generator used by estimator.
    """
    if mode is "infer":
      num_samples = len(test_samples)
    else:
      num_samples = self.config["data"][mode + "_num_samples"]

    batch_size = self.config[mode]["batch_size"]
    epochs = self.config[mode]["epochs"]
    max_steps = (num_samples * epochs // batch_size)

    samples = self.get_samples_fn(mode, test_samples)

    dataset = tf.data.Dataset.from_tensor_slices(samples)

    if mode == "infer":
      dataset = dataset.map(
        lambda test_sample: self.parse_fn(test_sample, mode),
        num_parallel_calls=4)
    else:
      dataset = dataset.flat_map(tf.data.TFRecordDataset)
      dataset = dataset.map(
        lambda raw_record: self.parse_fn_tfrecord(raw_record, mode),
        num_parallel_calls=4)

      if mode == "train":
        dataset = \
         dataset.shuffle(buffer_size=2000)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together
    dataset = dataset.repeat(epochs)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Add this to control the length of experiment by total_examples
    # Useful for hyper-parameter search with shorter experiments
    dataset = dataset.take(max_steps)

    # Add prefetch
    dataset = dataset.prefetch(self.config["run_config"]["prefetch"])

    # Create iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

  def get_samples_fn(self, mode, test_samples):
    """Collecting all samples in a dataset
    Returns a tuple of lists (images_path, labels)
    """
    data_dir = self.config["data"]["dir"]
    if mode == "infer":
      filenames = test_samples
    else:
      assert os.path.exists(data_dir), (
        "Cannot find " + data_dir)

      # return the list of filenames
      filename = []
      if mode == "train":
        filenames = [os.path.join(data_dir, 'train-%05d-of-%05d' % (i, self.config["data"]["train_num_records"]))
                     for i in range(self.config["data"]["train_num_records"])]
      elif mode == "eval":
        filenames = [os.path.join(data_dir, 'validation-%05d-of-%05d' % (i, self.config["data"]["eval_num_records"]))
                     for i in range(self.config["data"]["eval_num_records"])]
      else:
        assert False, "Unknow mode {} in get_samples_fn.".format(mode)
    return filenames

  def parse_fn_tfrecord(self, raw_record, mode):
    """Parse a single input sample
    """
    is_training = mode == "train"

    keys_to_features = {
      'image/encoded':
      tf.FixedLenFeature((), dtype=tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), dtype=tf.string, default_value='jpeg'),
      'image/class/label':
      tf.FixedLenFeature((), dtype=tf.int64, default_value=-1),
      'image/class/text':
      tf.FixedLenFeature((), dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
      tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
      tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
      tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
      tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
      tf.VarLenFeature(dtype=tf.int64),
    }

    # decode the record
    record = tf.parse_single_example(raw_record, keys_to_features)

    # create data iterms
    image = tf.image.decode_image(
      tf.reshape(record['image/encoded'], shape=[]),
      self.config['data']['depth'])

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # optinonally, preprocess the iterm
    image = self.augmenter(image,
                           self.config["data"]["height"],
                           self.config["data"]["width"],
                           is_training)

    label = tf.cast(
      tf.reshape(record['image/class/label'], shape=[]),
      dtype=tf.int32)

    return image, tf.one_hot(label, self.config['data']['num_classes'])

  def parse_fn(self, test_sample, mode):
    image = tf.read_file(test_sample)
    image = tf.image.decode_jpeg(image,
                                 channels=self.config["data"]["depth"],
                                 dct_method="INTEGER_ACCURATE")
    is_training = (mode == "train")
    image = self.augmenter(image,
                           self.config["data"]["height"],
                           self.config["data"]["width"],
                           is_training)    
    label = tf.one_hot(0, depth=self.config["data"]["num_classes"])

    return image, label

def build(config):
  """Returns the constructor of the inputter
  """
  return Inputter(config)
