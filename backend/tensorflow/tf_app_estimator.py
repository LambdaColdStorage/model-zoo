"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Implement TF application use estimator API
"""
from __future__ import print_function

import tensorflow as tf

from backend.tensorflow import tf_app


class TF_App_Estimator(tf_app.TF_App):
  def __init__(self, config):
    super(TF_App_Estimator, self).__init__(config)

    self.run_config = tf.estimator.RunConfig(
      session_config=self.session_config,
      model_dir=self.config['model']['dir'],
      save_summary_steps=self.config["run_config"]['save_summary_steps'],
      save_checkpoints_steps=self.config["run_config"]['save_checkpoints_steps'],
      keep_checkpoint_max=self.config["run_config"]['keep_checkpoint_max'],
      log_step_count_steps=self.config["run_config"]['log_every_n_iter'],
      train_distribute=None)

    devices = []
    for i in range(self.config['run_config']['num_gpu']):
      devices.append(u'/device:GPU:' + str(i))

    model_fn = (tf.contrib.estimator.replicate_model_fn(
                self.model_fn,
                loss_reduction=tf.losses.Reduction.MEAN,
                devices=devices))

    self.estimator = (tf.estimator.Estimator(model_fn=model_fn,
                                          config=self.run_config))

  def create_train_op_fn(self, mode, loss):
    """Create training operator
    Returns:
      A train_op used by estimator.
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
      # Setup global step
      global_step = tf.train.get_or_create_global_step()

      tf.identity(global_step, 'step')

      # Setup learning rate
      learning_rate = self.create_learning_rate_fn(global_step)

      # Setup optimizer
      optimizer = self.create_optimizer(learning_rate)

      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

      # Setup train_op
      train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      if 'trainable_var_list' in self.config['train']:
        train_vars = [v for v in train_vars
                      if any(x in v.name
                             for x in
                             self.config['train']['trainable_var_list'])]

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      # Use tf.group to avoid overhead in starting the training
      minimize_op = optimizer.minimize(loss,
                                       global_step,
                                       var_list=train_vars)
      train_op = tf.group(minimize_op, update_ops)

      return train_op
    else:
      return None

  def model_fn(self, features, labels, mode):
    """implementation of the model function
    Returns:
      A estimator spec used by estimator.
    """

    logits, predictions = self.modeler.create_graph_fn(mode, features)

    if mode == tf.estimator.ModeKeys.PREDICT:
      # Early return if in PREDICT mode
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      # Restore from ckpt in Train if self.config['train']['restore_ckpt'] is not none
      # This is better than warmstart, which does not restore moving
      # statistics for batchnorm
      if "restore_ckpt" in self.config['train']:
        variables_to_restore = {v.name.split(':')[0]: v
                                for v in tf.get_collection(
                                    tf.GraphKeys.GLOBAL_VARIABLES)}
        if self.config['train']['skip_restore_var_list'] is not None:
          variables_to_restore = {v: v for v in variables_to_restore
                                  if not any(x in v for x in
                                             self.config['train']['skip_restore_var_list'])}
        tf.train.init_from_checkpoint(self.config['train']['restore_ckpt'],
                                      variables_to_restore)

    # Create the loss
    loss = self.modeler.create_loss_fn(logits, labels)

    # Create the eval metrics
    metrics = self.modeler.create_eval_metrics_fn_estimator(predictions, labels)

    # Create the training operator
    train_op = self.create_train_op_fn(mode, loss)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

  def train(self):
    """Training interface
    """
    max_steps = (self.config["data"]["train_num_samples"] *
                 self.config["train"]["epochs"] //
                 self.config["train"]["batch_size"])

    (self.estimator.train(
     input_fn=lambda: self.inputter.input_fn(tf.estimator.ModeKeys.TRAIN),
     max_steps=max_steps,
     hooks=self.modeler.create_hooks_fn_estimator()))

  def eval(self):
    """Evaluation interface
    """
    eval_results = (self.estimator.evaluate(
      input_fn=lambda: self.inputter.input_fn(tf.estimator.ModeKeys.EVAL)))


  def infer(self, test_samples):
    """Inference interface
    """
    predictions = self.estimator.predict(
      input_fn=lambda: self.inputter.input_fn(tf.estimator.ModeKeys.PREDICT,
                                              test_samples=test_samples))

    for prediction, sample in zip(predictions, test_samples):
      self.modeler.display_prediction_estimator(prediction, sample)


  def inspect(self):
    """Inspect interface
    """
    pass


def build(config):
  """Returns the constructor of the application
  """
  return TF_App_Estimator(config)
