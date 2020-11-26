# Python

# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

"""Script to split titanic dataset into train/val"""
import argparse
import datetime
import logging
import os
import pandas as pd
import sklearn.model_selection

def parse_args():
  '''Parse command line arguments.'''
  parser = argparse.ArgumentParser(
      description='Split Dataset Titanic KFP Component')
  parser.add_argument('--input_data', type=str, required=True,
      help='GCS Path to input dataset, filename included')
  parser.add_argument('--output_path', type=str, required=True,
      help="GCS Path to output train and val files.")
  parser.add_argument('--log_level', type=str, default="INFO",
      help="Logging level.")
  return parser.parse_args()

def write_file(path: str, content: str):
  """Write string into local file.
  Args:
    path: the local path to the file.
    content: the string content to dump.
  """
  directory = os.path.dirname(path)
  if not os.path.exists(directory):
    os.makedirs(directory)
  elif os.path.exists(path):
    logging.warning('The file %s will be overwritten.', path)
  with open(path, 'w') as f:
    f.write(content)

def get_output_filenames(output_path: str):
  """Returns a dict of output filenames."""
  now = datetime.datetime.now()
  now_string = now.strftime("%Y%m%d_%H%M%S")
  filenames ={
      'train': os.path.join(output_path, "train_split_"+now_string+".csv"),
      'val': os.path.join(output_path, "val_split_"+now_string+".csv")
  }
  write_file("/tmp/train.txt", filenames['train'])
  write_file("/tmp/val.txt", filenames['val'])
  return filenames

def split_data(cl_args: argparse.ArgumentParser):
  """Entrypoint."""
  filenames = get_output_filenames(cl_args.output_path)
  df = pd.read_csv(cl_args.input_data, index_col=0)
  train, val = sklearn.model_selection.train_test_split(df, test_size=0.2,
      stratify=df['Survived'])
  train.to_csv(filenames['train'])
  val.to_csv(filenames['val'])

if __name__== "__main__":
  args = parse_args()
  logging.getLogger().setLevel(args.log_level)
  split_data(args)
