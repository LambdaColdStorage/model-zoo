"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

A TF application has a modeler and an inputter
"""
from __future__ import print_function
import time
import os

import tensorflow as tf

from backend.tensorflow_backend import tf_app


PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device="/cpu:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device
    return _assign


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def average_losses(tower_loss):
  return tf.reduce_mean(tower_loss)


def average_accuracies(tower_accuracies):
  return tf.reduce_mean(tower_accuracies)


class TF_App_Simple(tf_app.TF_App):
  def __init__(self, config):
    super(TF_App_Simple, self).__init__(config)

  def train(self):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    if not os.path.isdir(self.config["model"]["dir"]):
      tf.logging.info("Creating model directory %s",
                      self.config["model"]["dir"])
      os.makedirs(self.config["model"]["dir"])

    bs_per_gpu = self.config["train"]["batch_size_per_gpu"]
    save_summary_steps = self.config["run_config"]["save_summary_steps"]
    save_checkpoints_steps = \
        self.config["run_config"]["save_checkpoints_steps"]
    num_gpu = self.config['run_config']['num_gpu']

    max_steps = (self.config["data"]["train_num_samples"] *
                 self.config["train"]["epochs"] //
                 (bs_per_gpu * num_gpu))

    # Build training graph
    with tf.device("/cpu:0"):
      global_step = tf.train.get_or_create_global_step()
      learning_rate = self.modeler.create_learning_rate_fn(global_step)

      batch = self.inputter.input_fn("train")
      tower_losses = []
      tower_grads = []
      variables_to_restore = []

      for i in range(num_gpu):
        with tf.device(assign_to_device("/gpu:{}".format(i),
                       ps_device="/cpu:0")):
          _x = batch[0][i * bs_per_gpu:(i + 1) * bs_per_gpu]
          _y = batch[1][i * bs_per_gpu:(i + 1) * bs_per_gpu]

          logits, predictions = self.modeler.create_graph_fn("train", _x)

          # Initialize variables from a pre-trained ckpt
          if "restore_ckpt" in self.config["train"]:
            variables_to_restore = {v.name.split(':')[0]: v
                                    for v in tf.get_collection(
                                        tf.GraphKeys.GLOBAL_VARIABLES)}

            if ("skip_restore_var_list" in
                    self.config["train"]["skip_restore_var_list"]):
              variables_to_restore = {
                v: variables_to_restore[v] for
                v in variables_to_restore if not
                any(x in v for
                    x in self.config['train']['skip_restore_var_list'])}

          # Pin the trainable variables
          train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
          if "trainable_var_list" in self.config["train"]:
            train_vars = [v for v in train_vars
                          if any(x in v.name
                                 for x in
                                 self.config["train"]["trainable_var_list"])]

          # Compute per-gpu loss and gradient
          loss = self.modeler.create_loss_fn(logits, _y)

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

          grads = optimizer.compute_gradients(loss, var_list=train_vars)

          tower_losses.append(loss)
          tower_grads.append(grads)

          if i == 0:
            # Compute training accuracy from the first GPU
            training_accuracy = \
              self.modeler.create_eval_metrics_fn(predictions, _y)
            tf.summary.scalar("training_accuracy", training_accuracy)

      # Compute average loss and gradient
      tower_losses = average_losses(tower_losses)
      tower_grads = average_gradients(tower_grads)

      # # Create train_op to minize the loss
      minimize_op = optimizer.apply_gradients(tower_grads,
                                              global_step=global_step)

      # # Force moving statistics to be updated during training
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = tf.group(minimize_op, update_ops)

      # Create summary writer
      tf.summary.scalar("train_loss", tower_losses)
      summary_writer = tf.summary.FileWriter(self.config["model"]["dir"],
                                             graph=tf.get_default_graph())
      merged_summary_op = tf.summary.merge_all()

      if variables_to_restore:
        saver_pre_trained = tf.train.Saver(
          var_list=variables_to_restore)

      saver = tf.train.Saver(
        max_to_keep=self.config["run_config"]["keep_checkpoint_max"])

    # Run training
    with tf.Session(config=self.session_config) as sess:
      if variables_to_restore and not tf.train.checkpoint_exists(
        self.config["model"]["dir"] + "/model.*"):

        start = time.time()
        if tf.train.checkpoint_exists(self.config["train"]["restore_ckpt"] +
                                      "/*.ckpt"):
          saver_pre_trained.restore(sess,
                                    tf.train.latest_checkpoint(
                                      self.config["train"]["restore_ckpt"]))
        else:
          raise ValueError("Cannot find pre-trained model to restore from")
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var)
                                       for var in global_vars])
        not_initialized_vars = [v for (v, f) in
                                zip(global_vars, is_not_initialized) if not f]
        init_op = tf.initialize_variables(not_initialized_vars)
        sess.run(init_op)
        end = time.time()
        print("Restored parameters from " +
              self.config["train"]["restore_ckpt"] +
              " in " + str(end - start) + "sec.")
      elif tf.train.checkpoint_exists(
        self.config["model"]["dir"] + "/model.*"):
        saver.restore(sess,
                      tf.train.latest_checkpoint(
                        self.config["model"]["dir"]))
      else:
        print("Initialize globla variables ... ")
        sess.run(tf.global_variables_initializer())

      _global_step = sess.run(global_step)

      if _global_step >= max_steps:
        print("Training has already reached the maximum steps.")
      else:
        if _global_step == 0:
          print("Start training from step " + str(_global_step))
        else:
          print("Resume training from step " + str(_global_step))

        while _global_step < max_steps:
          _, _loss, _summary, _global_step, _training_accuracy = sess.run(
            [train_op, tower_losses,
             merged_summary_op, global_step, training_accuracy])

          if _global_step % self.config["run_config"]["log_every_n_iter"] == 0:
            print("Step " + str(_global_step) +
                  ": training accuracy " + str(_training_accuracy))

          if _global_step % save_summary_steps == 0:
            summary_writer.add_summary(_summary, _global_step)

          if _global_step % save_checkpoints_steps == 0:
            save_path = saver.save(
              sess,
              self.config["model"]["dir"] + "/model.ckpt",
              global_step=_global_step)
            print("Saving checkpoint " + save_path)

        if max_steps % save_checkpoints_steps != 0:
          print("Saving checkpoint for the last step ...")
          save_path = saver.save(sess,
                                 self.config["model"]["dir"] + "/model.ckpt",
                                 global_step=_global_step)
          print("Checkpoint " + save_path + " has been saved.")
        if max_steps % save_summary_steps != 0:
            summary_writer.add_summary(_summary, _global_step)

      summary_writer.flush()
      summary_writer.close()

  def eval(self):
    """Evaluation interface
    """
    pass

  def train_and_eval(self):
    """Evaluation interface
    """
    pass

  def infer(self):
    """Inference interface
    """
    pass

  def inspect(self):
    """Inspect interface
    """
    pass

  def tune(self):
    """Inspect interface
    """
    pass


def build(config):
  """Returns the constructor of the application
  """
  return TF_App_Simple(config)
