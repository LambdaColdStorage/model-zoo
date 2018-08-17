"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

The Base class for all applications.
"""

from __future__ import print_function
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class App(object):
  """Base class for all applications
  """

  def __init__(self, config):
    self.config = config

  """Application interfaces
  """

  @abc.abstractmethod
  def train(self):
    """Training interface
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def eval(self):
    """Evaluation interface
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def train_and_eval(self):
    """Evaluation interface
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def infer(self):
    """Inference interface
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def inspect(self):
    """Inspect interface
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def tune(self):
    """Inspect interface
    """
    raise NotImplementedError()
