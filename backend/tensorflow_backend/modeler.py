"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Define model interfaces for TF backend
"""
from __future__ import print_function
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Modeler(object):
  def __init__(self, config):
    self.config = config

  """Model interfaces for TF backend
  """
  @abc.abstractmethod
  def create_graph_fn(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_loss_fn(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_learning_rate_fn(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_eval_metrics_fn(self, *argv):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_train_op_fn(self, *argv):
    raise NotImplementedError()
