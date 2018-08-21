from backend.tensorflow.augmenters.external import inception_preprocessing


def augment(image, output_height, output_width, is_training=False):
  return inception_preprocessing.preprocess_image(
    image, output_height, output_width, is_training)
