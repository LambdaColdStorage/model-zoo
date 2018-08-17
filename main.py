"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Main python API for running machine learning jobs
"""

import argparse
import importlib
import glob
import os
import yaml
import fileinput


def parse_config(config_path):
  """Parse a config file into a config object.
  """
  with open(config_path) as file:
    config = yaml.load(file.read())

  return config


def parse_test(args):
  """Parse arguments into a list of samples.
  """
  # TODO:
  # Add support for bash redirection
  samples = []
  if args.mode == 'infer':
    if not args.test_path and not args.test_samples:
      assert False, "Please provide test_path or test_samples."

    if args.test_path:
      # A single file where each line is a path to a test input
      # TODO:
      # Make this more general so each line can be a sample (for nmt etc.)
      for line in fileinput.input(os.path.expanduser(args.test_path)):
        clean = line.strip()
        if clean:
          samples.append(clean)
    else:
      # Strip the white space from the samples
      samples = [x for x in args.test_samples.split(',')]
  return samples


def main():
  fmt_cls = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=fmt_cls)

  parser.add_argument('config_path',
                      type=str,
                      help='Path to a config file. The config file should '
                      'provide all the information that is needed to run a '
                      'application.')
  parser.add_argument('mode', choices=['train', 'eval', 'train_and_eval',
                                       'infer', 'tune', 'inspect'],
                      type=str,
                      help='Choose a job mode from train, eval, '
                      'train_and_eval, infer and inspect.')
  parser.add_argument('--test_path',
                      type=str,
                      help='The path of a testing file. '
                      'Each line in this file is a testing sample. '
                      'Only used in the infer mode.')
  parser.add_argument('--test_samples',
                      help='A string of comma seperated testing data. '
                      'Only used (also must be provided) if '
                      'test_path is not provided in the infer mode.')
  args = parser.parse_args()

  # Load config
  config = parse_config(args.config_path)

  # Build an application
  app = importlib.import_module(config['app']).build(config)

  if args.mode == 'train':
    app.train()
  elif args.mode == 'eval':
    app.eval()
  elif args.mode == 'train_and_eval':
    app.train_and_eval()
  elif args.mode == 'infer':
    test_samples = parse_test(args)
    app.infer(test_samples)
  elif args.mode == 'tune':
    app.tune()
  elif args.mode == 'inspect':
    app.inspect()


if __name__ == '__main__':
  main()
