"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF inputter interfaces for image segmentation (real data)
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
      meta_filename = None
    else:
      num_samples = self.config["data"][mode + "_num_samples"]
      meta_filename = (self.config["data"]["dir"] + "/" +
                       self.config["data"][mode + "_meta"])

    batch_size = self.config[mode]["batch_size"]
    epochs = self.config[mode]["epochs"]
    max_steps = (num_samples * epochs // batch_size)

    samples = self.get_samples_fn(mode, meta_filename, test_samples)

    dataset = tf.data.Dataset.from_tensor_slices(samples)

    if mode == "train":
     dataset = \
       dataset.shuffle(buffer_size=self.config["data"]["train_num_samples"])

    # Call repeat after shuffling, rather than before, to prevent separate
    # epochs blurred boundaries
    dataset = dataset.repeat(epochs)

    dataset = dataset.map(
      lambda image, label: self.parse_fn(mode, image, label),
      num_parallel_calls=4)

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

  def get_samples_fn(self, mode, meta_filename, test_samples):
    """Collecting all samples in a dataset
    Returns a tuple of lists (images_path, labels)
    """
    if mode == "infer":
      images_path = test_samples
      labels_path = test_samples
      return (images_path, labels_path)
    elif mode == "train" or \
            mode == "eval":
      assert os.path.exists(meta_filename), (
        "Cannot find " + meta_filename)

      images_path = []
      labels_path = []
      with open(meta_filename) as f:
        parsed = csv.reader(f, delimiter=",", quotechar="'")
        for row in parsed:
          images_path.append(self.config["data"]["dir"] + "/" + row[0])
          labels_path.append(self.config['data']['dir'] + '/' + row[1])
      return (images_path, labels_path)

    else:
      assert False, "Unknow image segmentation inputter mode: {}".forma(mode)

  def parse_fn(self, mode, image_path, label_path):
    """Parse a single input sample
    """
    image = tf.read_file(image_path)
    image = tf.image.decode_png(image, channels=self.config['data']['depth'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if mode == "infer":
      image = image - 0.5
      label = image
      return image, label
    else:
      label = tf.read_file(label_path)
      label = tf.image.decode_png(label, channels=1)
      label = tf.cast(label, dtype=tf.int64)

      is_training = (mode == "train")
      return self.augmenter(image, label,
                            self.config["data"]["output_height"],
                            self.config["data"]["output_width"],
                            self.config["data"]["resize_side_min"],
                            self.config["data"]["resize_side_max"],
                            is_training=is_training)


def build(config):
  """Returns the constructor of the inputter
  """
  return Inputter(config)
