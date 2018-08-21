import importlib

networks = {
  "resnet32_slim": "backend.tensorflow.networks.resnet32_slim",
  "resnet50_slim": "backend.tensorflow.networks.resnet50_slim",
  "nasnet_A_large_slim": "backend.tensorflow.networks.nasnet_A_large_slim",
  "fcn": "backend.tensorflow.networks.fcn"
}


def get_network(name):
  return getattr(importlib.import_module(networks[name]), "net")
