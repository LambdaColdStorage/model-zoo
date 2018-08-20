"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF modeler interfaces for image segmentation.
"""
from __future__ import print_function
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

from backend.tensorflow.modelers import modeler


class Modeler(modeler.Modeler):
  def __init__(self, config):
    super(Modeler, self).__init__(config)
    # create a color palette
    self.colors = np.random.randint(255,
                                    size=(config["data"]["num_classes"], 3))

  def create_loss_fn(self, logits, labels):
    """Create loss operator
    Returns:
      loss
    """
    logits = tf.reshape(logits, [-1, self.config['data']['num_classes']])
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)

    loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    l2_var_list = [v for v in tf.trainable_variables()]
    if "skip_l2_var_list" in self.config["train"]:
      l2_var_list = [v for v in l2_var_list
                     if not any(x in v.name for
                                x in self.config["train"]["skip_l2_var_list"])]

    loss_l2 = self.config["train"]["l2_weight_decay"] * tf.add_n(
      [tf.nn.l2_loss(v) for v in l2_var_list])

    loss = tf.identity(loss_cross_entropy + loss_l2, "total_loss")

    return loss

  def create_eval_metrics_fn(self, predictions, labels):
    """ Create the evaluation metric
    Returns:
      A dictionary of metrics used by estimator.
    """
    equality = tf.equal(predictions["classes"],
                        labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy

  def create_eval_metrics_fn_estimator(self, predictions, labels):
    """ Create the evaluation metric
    Returns:
      A dictionary of metrics used by estimator.
    """
    accuracy = tf.metrics.accuracy(
      predictions["classes"],
      labels)

    # for logging in the console during training
    tf.identity(accuracy[1], name="running_accuracy")

    # for tensorboard train_accuracy
    tf.summary.scalar("train_accuracy", accuracy[1])

    # for tensorboard eval_accuracy (only run by estimator.eval)
    metrics = {"eval_accuracy": accuracy}

    return metrics

  def create_hooks_fn_estimator(self):
    """Returns a list of training hooks
    """
    tensors_to_log = {
      "running_accuracy": "running_accuracy",
      "total_loss": "total_loss",
      "step": "step"}

    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=self.config["run_config"]["log_every_n_iter"])
    return [logging_hook]

  def create_graph_fn(self, mode, inputs):
    """Create forward graph
    Returns:
      logits: A tensorf holds the raw outputs of the network.
      predictions: A dictionary hold post-processed results.
    """
    is_training = (mode == "train")
    num_classes = self.config["data"]["num_classes"]

    return self.net(inputs, num_classes, is_training=is_training)

  def render_label(self, label, num_classes, label_colors):

    label = label.astype(int)
    r = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    g = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    b = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)

    for i_color in range(0, num_classes):
      r[label == i_color] = label_colors[i_color, 0]
      g[label == i_color] = label_colors[i_color, 1]
      b[label == i_color] = label_colors[i_color, 2]

    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

  def display_prediction_simple(self, predictions, sample=None):
    """Displays predcition result for simple API
    Input:
    prediction: a list of dictionary. Each dictionary has a probabilities field
                and a classes field.
    Each of these fields is also a list of results.
    """
    for prediction in predictions:
      classes = prediction["classes"]
      for c in classes:
        render_label = self.render_label(c,
                                         self.config['data']['num_classes'],
                                         self.colors)
        image_out = Image.fromarray(render_label, 'RGB')
        plt.imshow(image_out)
        plt.show()

  def display_prediction_estimator(self, prediction, sample=None):
    """Displays predcition result for estimator API
    Input:
    prediction: a dictinary that has a probabilities field and a classes field.
    Each of these fields represent a single result.
    """
    predicted_label = prediction['classes']
    render_label = self.render_label(predicted_label,
                                     self.config['data']['num_classes'],
                                     self.colors)

    image_out = Image.fromarray(render_label, 'RGB')

    plt.imshow(image_out)
    plt.show()


def build(config):
  """Returns the constructor of the modeler
  """
  return Modeler(config)
