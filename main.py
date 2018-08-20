"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Main python API for running machine learning jobs
"""

import argparse
import importlib

import app_parser


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
  config = app_parser.parse_config(args.config_path)

  # Build an application
  app = importlib.import_module("backend." + config["backend"] + "." + config["app"]).build(config)

  app.run(args)


if __name__ == '__main__':
  main()
