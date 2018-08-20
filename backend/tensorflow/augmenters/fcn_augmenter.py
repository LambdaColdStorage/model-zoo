import tensorflow as tf
from backend.tensorflow.augmenters.external import image_utils

def augment(image, output_height, output_width,
            resize_side_min, resize_side_max, is_training=False):
  if is_training:
    resize_side = tf.random_uniform(
      [],
      minval=resize_side_min,
      maxval=resize_side_max + 1,
      dtype=tf.int32)

    image = image_utils.aspect_preserving_resize(
      image, esize_side,
      self.config['data']['depth'],
      'bilinear')
    image = image_utils.central_crop(
      [image],
      self.config['data']['output_height'],
      self.config['data']['output_width'])[0]
    image.set_shape(
      [self.config['data']['output_height'],
       self.config['data']['output_width'],
       self.config['data']['depth']])
    label = image_utils.aspect_preserving_resize(
      label,
      resize_side,
      1,
      'nearest')
    label = image_utils.central_crop(
      [label],
      self.config['data']['output_height'],
      self.config['data']['output_width'])[0]
    label.set_shape(
      [self.config['data']['output_height'],
       self.config['data']['output_width'],
       1])
  else:
    image = tf.image.resize_images(
      image,
      [self.config['data']['output_height'],
       self.config['data']['output_width']],
      tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize_images(
      label,
      [self.config['data']['output_height'],
       self.config['data']['output_width']],
      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = image - 0.5
  return image, label
