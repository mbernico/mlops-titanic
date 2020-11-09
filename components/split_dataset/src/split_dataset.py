# Python

# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

"""Script to split titanic dataset into train/val"""

import argparse
import logging
import pandas as pd
import sklearn.model_selection

def parse_args():
  '''Parse command line arguments.'''
  parser = argparse.ArgumentParser(
      description='Split Dataset Titanic KFP Component')
  parser.add_argument('--input_data', type=str, required=True,
      help='GCS Path to input dataset.')
  parser.add_argument('--train_output', type=str, required=True,
      help="GCS Path to training output dataset.")
  parser.add_argument('--val_output', type=str, required=True,
      help="GCS Path to training output dataset.")
  parser.add_argument('--log_level', type=str, default="INFO",
      help="Logging level.")
  return parser.parse_args()

def main():
  """Entrypoint."""
  args = parse_args()
  logging.getLogger().setLevel(args.log_level)
  df = pd.read_csv(args.input_data, index_col=0)
  train, val = sklearn.model_selection.train_test_split(df, test_size=0.2,
      stratify=df['Survived'])
  train.to_csv(args.train_output)
  val.to_csv(args.val_output)

if __name__== "__main__":
  main()
