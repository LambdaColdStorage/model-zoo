import tensorflow as tf

from backend.tensorflow_backend.networks.external.tf_slim import resnet_v2

slim = tf.contrib.slim

def net(inputs, num_classes, is_training):
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    return resnet_v2.resnet_v2_50(inputs,
                                  num_classes,
                                  is_training=is_training)
