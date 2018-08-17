"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF inputter interfaces for image classification (synthetic data)
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest

from backend.tensorflow_backend import inputter


class Inputter(inputter.Inputter):
  def __init__(self, config):
    super(Inputter, self).__init__(config)

  def input_fn(self, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
      batch_size = self.config["train"]["batch_size"]
    else:
      assert False, \
        "Unknown mode for image_classifcation_inputter_syn: '{}'".format(mode)

    max_steps = (self.config["data"]["train_num_samples"] *
                 self.config["train"]["epochs"] //
                 batch_size)

    input_dtype = tf.float32
    label_dtype = tf.int32
    input_value = 0.0
    label_value = 0

    input_shape = tf.TensorShape([batch_size,
                                  self.config["data"]["height"],
                                  self.config["data"]["width"],
                                  self.config["data"]["depth"]])
    label_shape = tf.TensorShape([batch_size,
                                  self.config["data"]["num_classes"]])

    input_element = nest.map_structure(
        lambda s: tf.constant(input_value, input_dtype, s), input_shape)

    label_element = nest.map_structure(
        lambda s: tf.constant(label_value, label_dtype, s), label_shape)

    element = (input_element, label_element)

    dataset = tf.data.Dataset.from_tensors(element).repeat(max_steps)

    # Add prefetch
    dataset = dataset.prefetch(self.config["data"]["prefetch"])

    # Create iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

  def get_samples_fn(self, *argv):
    pass

  def preprocessing(self, *argv):
    pass

  def parse_fn(self, mode, *argv):
    pass


def build(config):
  """Returns the constructor of the inputter
  """
  return Inputter(config)
