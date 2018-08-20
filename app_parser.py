import yaml
import fileinput
import os


def parse_config(config_path):
  """Parse a config file into a config object.
  """
  with open(config_path) as file:
    config = yaml.load(file.read())
  config["train"]["batch_size"] = config["train"]["batch_size_per_gpu"] * \
      config["run_config"]["num_gpu"]
  config["eval"]["batch_size"] = config["eval"]["batch_size_per_gpu"] * \
      config["run_config"]["num_gpu"]
  config["infer"]["batch_size"] = config["infer"]["batch_size_per_gpu"] * \
      config["run_config"]["num_gpu"]
  return config


def parse_test(args):
  """Parse arguments into a list of samples.
  """
  # TODO:
  # Add support for bash redirection
  samples = []
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
    samples = [x for x in args.test_samples.split(",")]
  return samples
