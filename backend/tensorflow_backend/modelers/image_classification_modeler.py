"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF modeler interfaces for image classification.
"""
from __future__ import print_function

import tensorflow as tf

from backend.tensorflow_backend import modeler
from backend.tensorflow_backend.networks import network_factory

# from external.tf_slim import resnet_v2


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
      logits, predictions
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


def build(config):
  """Returns the constructor of the modeler
  """
  return Modeler(config)
