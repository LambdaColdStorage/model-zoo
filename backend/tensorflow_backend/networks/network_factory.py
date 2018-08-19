import importlib

networks = {
  "resnet32slim": "backend.tensorflow_backend.networks.resnet32slim",
  "resnet50slim": "backend.tensorflow_backend.networks.resnet50slim"
}

def get_network(name):
  return getattr(importlib.import_module(networks[name]), "net")
