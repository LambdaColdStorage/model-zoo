"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

The Base class for all applications.
"""

from __future__ import print_function
import abc
import six

import app_parser


@six.add_metaclass(abc.ABCMeta)
class App(object):
  """Base class for all applications
  """

  def __init__(self, config):
    self.config = config

  """Application interfaces
  """

  def run(self, args):
    if args.mode == 'train':
      self.train()
    elif args.mode == 'eval':
      self.eval()
    elif args.mode == 'train_and_eval':
      self.train_and_eval()
    elif args.mode == 'infer':
      test_samples = app_parser.parse_test(args)
      self.infer(test_samples)
    elif args.mode == 'tune':
      self.tune()
    elif args.mode == 'inspect':
      self.inspect()
    else:
      assert False, "Unknown mode : '{}'".format(args.mode)

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
    """Training and Evaluation interface
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
    """Hyper-parameter tuning interface
    """
    raise NotImplementedError()
