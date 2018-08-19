"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Class for TF application.
"""
from __future__ import print_function
import importlib
import random
import os

import tensorflow as tf

import app


class TF_App(app.App):
  def __init__(self, config):
    super(TF_App, self).__init__(config)

    # A TF application has a modeler and an inputter
    self.modeler = importlib.import_module(config['modeler']).build(config)
    self.inputter = importlib.import_module(config['inputter']).build(config)
    self.session_config = self.create_session_config()
    self.batch_size = self.config['train']['batch_size_per_gpu'] * \
        self.config['run_config']['num_gpu']
    tf.logging.set_verbosity(tf.logging.INFO)

  def create_session_config(self):
    """create session_config
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                                allow_growth=True)

    # set number of GPU devices
    device_count = {'GPU': self.config['run_config']['num_gpu']}

    session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=self.config['run_config']['log_device_placement'],
      device_count=device_count,
      gpu_options=gpu_options)

    return session_config

  def create_optimizer(self, learning_rate):
    # Setup optimizer
    if self.config["train"]["optimizer"] == "adadelta":
      optimizer = tf.train.AdadeltaOptimizer(
          learning_rate=learning_rate)
    elif self.config["train"]["optimizer"] == "adagrad":
      optimizer = tf.train.AdagradOptimizer(
          learning_rate=learning_rate)
    elif self.config["train"]["optimizer"] == "adam":
      optimizer = tf.train.AdamOptimizer(
          learning_rate=learning_rate)
    elif self.config["train"]["optimizer"] == "ftrl":
      optimizer = tf.train.FtrlOptimizer(
          learning_rate=learning_rate)
    elif self.config["train"]["optimizer"] == "momentum":
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=self.config["train"]["optimizer_momentum"],
          name="Momentum")
    elif self.config["train"]["optimizer"] == "rmsprop":
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=learning_rate)
    elif self.config["train"]["optimizer"] == "sgd":
      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    else:
      raise ValueError("Optimizer [%s] was not recognized" %
                       self.config["train"]["optimizer"])
    return optimizer

  def create_learning_rate_fn(self, global_step):
    """Create learning rate
    Returns:
      A learning rate calcualtor used by TF's optimizer.
    """
    if "piecewise_learning_rate_decay" in self.config["train"]:
      initial_learning_rate = self.config["train"]["learning_rate"]
      batches_per_epoch = (self.config["data"]["train_num_samples"] /
                           (self.config["train"]["batch_size_per_gpu"] *
                            self.config['run_config']['num_gpu']))
      boundaries = list(map(float,
                        self.config["train"]["piecewise_boundaries"].split(",")))
      boundaries = [int(batches_per_epoch * boundary) for boundary in boundaries]

      decays = list(map(float,
                    self.config["train"]["piecewise_learning_rate_decay"].split(",")))
      values = [initial_learning_rate * decay for decay in decays]

      learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    else:
      learning_rate = self.config["train"]["learning_rate"]

    tf.identity(learning_rate, name="learning_rate")
    tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate

  def train_and_eval(self):
    """Training and Evaluation interface
    """
    self.train()
    self.eval()

  def tune(self):
    """Hyper-parameter Tuning interface
    """
    def type_convert(s):
        ''' convert value to int, float or str'''
        try:
            float(s)

            tp = 1 if s.count('.') == 0 else 2
        except ValueError:
            tp = -1

        if tp == 1:
          return int(v)
        elif tp == 2:
          return float(v)
        elif tp == -1:
          return v
        else:
          assert False, "Unknown type for hyper parameter: '{}'".format(tp)

    # setup the tunning jobs
    num_trials = self.config['tune']['num_trials']

    # update the fixed parameters in self.config['train']
    for field in list(self.config['tune']['fixedparams'].keys()):
      for param in list(self.config['tune']['fixedparams'][field].keys()):
        self.config[field][param] = \
          self.config['tune']['fixedparams'][field][param]

    # randomly generate hyper parameters for each trail
    dir_ori = self.config['model']['dir'] + '/tune/tune'
    t = 0
    while t < num_trials:
      dir_update = dir_ori
      for field in list(self.config['tune']['hyperparams'].keys()):
        if field in self.config:
          for param in list(self.config['tune']['hyperparams'][field].keys()):
            if param in self.config[field]:
              values = \
                self.config['tune']['hyperparams'][field][param].split(',')
              v = random.choice(values)
              self.config[field][param] = type_convert(v)
              dir_update = dir_update + '_' + param + '_' + str(v)

      # run experiment the same hyper parameters haven't been used before
      if not os.path.isdir(dir_update):
        self.config['model']['dir'] = dir_update
        self.train_and_eval()
        # print(dir_update)
        t = t + 1