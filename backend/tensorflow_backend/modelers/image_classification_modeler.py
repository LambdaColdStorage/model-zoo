"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF modeler interfaces for image classification.
"""
from __future__ import print_function

import tensorflow as tf

from backend.tensorflow_backend import modeler


class Modeler(modeler.Modeler):
  def __init__(self, config):
    super(Modeler, self).__init__(config)

  def create_graph_fn(self, mode, inputs):
    """Create forward graph
    Returns:
      logits, predictions
    """
    pass

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

  def create_learning_rate_fn(self, global_step):
    """Create learning rate
    Returns:
      A learning rate calcualtor used by TF's optimizer.
    """
    if "piecewise_learning_rate_decay" in self.config["train"]:
      initial_learning_rate = self.config["train"]["learning_rate"]
      batches_per_epoch = (self.config["data"]["train_num_samples"] /
                           (self.config["train"]["batch_size_per_gpu"] *
                            self.config['run_config']['num_gpu']))
      boundaries = list(map(float,
                            self.config["train"]["piecewise_boundaries"].split(",")))
      boundaries = [int(batches_per_epoch * boundary) for boundary in boundaries]

      decays = list(map(float,
                        self.config["train"]["piecewise_learning_rate_decay"].split(",")))
      values = [initial_learning_rate * decay for decay in decays]

      learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    else:
      learning_rate = self.config["train"]["learning_rate"]

    tf.identity(learning_rate, name="learning_rate")
    tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate

  def create_eval_metrics_fn(self, predictions, labels):
    """ Create the evaluation metric
    Returns:
      A dictionary of metrics used by estimator.
    """
    equality = tf.equal(predictions["classes"],
                        tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy

  def create_train_op_fn(self, *argv):
    pass

  def create_graph_fn(self, *argv):
    pass
