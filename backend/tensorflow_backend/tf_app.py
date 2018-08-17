"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Class for TF application.
"""
from __future__ import print_function
import importlib
from tensorflow.python.client import device_lib

import tensorflow as tf

import app


def create_session_config(config):
  """create session_config
  """

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                              allow_growth=True)

  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=config['log_device_placement'],
    gpu_options=gpu_options)

  return session_config


class TF_App(app.App):
  def __init__(self, config):
    super(TF_App, self).__init__(config)

    # A TF application has a modeler and an inputter
    self.modeler = importlib.import_module(config['modeler']).build(config)
    self.inputter = importlib.import_module(config['inputter']).build(config)
    self.session_config = create_session_config(config['run_config'])
