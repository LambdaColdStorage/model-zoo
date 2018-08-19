"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement resnet interfaces based on TF-slim
"""
from __future__ import print_function
import importlib

import tensorflow as tf

from backend.tensorflow_backend.modelers import image_classification_modeler
from external.tf_slim import resnet_v2

slim = tf.contrib.slim

net_factory = {
  "resnet32": "resnet_v2_32",
  "resnet50": "resnet_v2_50"
}

class Modeler(image_classification_modeler.Modeler):
  def __init__(self, config):
    super(Modeler, self).__init__(config)

  def create_graph_fn(self, mode, inputs):
    """Create forward graph
    Returns:
      logits, predictions
    """
    is_training = (mode == "train")
    num_classes = self.config["data"]["num_classes"]
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):

      net_creator = getattr(resnet_v2, net_factory[self.config["model"]["name"]])

      logits, end_points = net_creator(inputs,
                                       num_classes,
                                       is_training=is_training)

    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["predictions"]
    }

    return logits, predictions


def build(config):
  """Returns the constructor of the modeler
  """
  return Modeler(config)
