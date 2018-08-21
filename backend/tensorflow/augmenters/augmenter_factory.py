import importlib

networks = {
  "cifar": "backend.tensorflow.augmenters.cifar_augmenter",
  "vgg": "backend.tensorflow.augmenters.vgg_augmenter",
  "inception": "backend.tensorflow.augmenters.inception_augmenter",
  "trivial": "backend.tensorflow.augmenters.trivial_augmenter",
  "fcn": "backend.tensorflow.augmenters.fcn_augmenter"
}


def get_augmenter(name):
  return getattr(importlib.import_module(networks[name]), "augment")
