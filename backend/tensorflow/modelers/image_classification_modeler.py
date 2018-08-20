"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF modeler interfaces for image classification.
"""
from __future__ import print_function
import numpy as np

import tensorflow as tf

from backend.tensorflow import modeler
from backend.tensorflow.networks import network_factory


class Modeler(modeler.Modeler):
  def __init__(self, config):
    super(Modeler, self).__init__(config)
    self.net = network_factory.get_network(self.config["network"])

  def create_loss_fn(self, logits, labels):
    """Create loss operator
    Returns:
      loss
    """
    loss_cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

    loss_l2 = self.config["train"]["l2_weight_decay"] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

    return loss

  def create_eval_metrics_fn(self, predictions, labels):
    """ Create the evaluation metric
    Returns:
      A dictionary of metrics used by estimator.
    """
    equality = tf.equal(predictions["classes"],
                        tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy

  def create_eval_metrics_fn_estimator(self, predictions, labels):
    """ Create the evaluation metric
    Returns:
      A dictionary of metrics used by estimator.
    """
    accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1),
      predictions['classes'])

    # for logging in the console during training
    tf.identity(accuracy[1], name='running_accuracy')

    # for tensorboard train_accuracy
    tf.summary.scalar('train_accuracy', accuracy[1])

    # for tensorboard eval_accuracy (only run by estimator.eval)
    metrics = {'eval_accuracy': accuracy}

    return metrics

  def create_hooks_fn_estimator(self):
    """Returns a list of training hooks
    """
    tensors_to_log = {
      'running_accuracy': 'running_accuracy',
      'total_loss': 'total_loss',
      'step': 'step'}

    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=self.config['run_config']['log_every_n_iter'])
    return [logging_hook]

  def create_graph_fn(self, mode, inputs):
    """Create forward graph
    Returns:
      logits: A tensorf holds the raw outputs of the network.
      predictions: A dictionary hold post-processed results.
    """
    is_training = (mode == "train")
    num_classes = self.config["data"]["num_classes"]

    logits, end_points = self.net(inputs,
                                  num_classes,
                                  is_training=is_training)

    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["predictions"]
    }

    return logits, predictions

  def display_prediction_simple(self, predictions, sample=None):
    """Displays predcition result for simple API
    Input:
    prediction: a list of dictionary. Each dictionary has a probabilities field and a classes field.
    Each of these fields is also a list of results.
    """
    for prediction in predictions:
      probabilities = prediction["probabilities"]
      classes = prediction["classes"]
      for p, c in zip(probabilities, classes):
        print("class: " + str(c) + ", probability: " + str(p[c]))

  def display_prediction_estimator(self, prediction, sample=None):
    """Displays predcition result for estimator API
    Input:
    prediction: a dictinary that has a probabilities field and a classes field.
    Each of these fields represent a single result.
    """
    print('class: ' +
          str(prediction['classes']) +
          ', probability: ' +
          str(prediction['probabilities'][prediction['classes']]))


def build(config):
  """Returns the constructor of the modeler
  """
  return Modeler(config)
