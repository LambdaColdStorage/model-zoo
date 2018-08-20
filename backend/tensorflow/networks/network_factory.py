import importlib

networks = {
  "resnet32slim": "backend.tensorflow.networks.resnet32slim",
  "resnet50slim": "backend.tensorflow.networks.resnet50slim",
  "fcn": "backend.tensorflow.networks.fcn"
}


def get_network(name):
  return getattr(importlib.import_module(networks[name]), "net")
