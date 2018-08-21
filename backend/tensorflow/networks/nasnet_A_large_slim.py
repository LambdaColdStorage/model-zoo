import tensorflow as tf

from backend.tensorflow.networks.external.tf_slim import nasnet

slim = tf.contrib.slim


def net(inputs, num_classes, is_training):
  with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
    logits, end_points = nasnet.build_nasnet_large(inputs,
                                                   num_classes,
                                                   is_training=is_training)
    predictions = {
      "classes": tf.argmax(logits, axis=1),
      "probabilities": end_points["Predictions"]
    }

    return logits, predictions
