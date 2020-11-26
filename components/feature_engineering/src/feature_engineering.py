# Python

# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

"""Script to create features for Titanic."""

from typing import List

import argparse
import datetime
import logging
import os
import pandas as pd

from imputer import MissingAgeImputer

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Drops unused columns."""
  return df.drop(columns=['PassengerId'], inplace=True)

def create_title_feature(df: pd.DataFrame) -> pd.DataFrame:
  """Creates a new column Title that extracts title from name."""
  df['Title'] = df.Name.str.split(',', expand=True)[1].str.split(
      '.', expand=True)[0].str.lstrip()
  df['Title'] = df['Title'].replace(['Miss', 'Ms', 'Mlle'],
      'Unmarried Feminine Title')
  df['Title'] = df['Title'].replace(['Mrs','Mme'], 'Married Feminine Title')
  df['Title'] = df['Title'].replace(
      ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Rev',], 'Military/Clergy')
  df['Title'] = df['Title'].replace(['Don', 'Sir'], 'Masculine Noble')
  df['Title'] = df['Title'].replace(['Lady', 'Mme', 'the Countess', 'Dona'],
  'Feminine Noble')
  df.drop(columns=['Name'], inplace=True)
  return df

def fill_embarked(df):
  """Fill missing embarked"""
  df['Embarked'] = df['Embarked'].fillna('S')
  return df

def create_deck_feature(df: pd.DataFrame) -> pd.DataFrame:
  """Creates a new Deck feature"""
  df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
  # Passenger in the T deck is changed to A
  idx = df[df['Deck'] == 'T'].index
  df.loc[idx, 'Deck'] = 'A'
  df.drop(columns=['Cabin'], inplace=True)
  return df

def transform_dataset(df:pd.DataFrame, transformations: List) -> pd.DataFrame:
  '''Performs all transformations on dataset.'''
  # pylint: disable=expression-not-assigned
  [t(df) for t in transformations]
  return df

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
  '''Creates bins based on total family size.
    May want to bin this...
  '''
  df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
  return df

def create_ticket_freq(df: pd.DataFrame) -> pd.DataFrame:
  '''Create Ticket frequency distribution feature.'''
  df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
  df.drop(columns=['Ticket'], inplace=True)
  return df

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
      'train': os.path.join(output_path, "train_fe_"+now_string+".csv"),
      'val': os.path.join(output_path, "val_fet_"+now_string+".csv"),
      'test': os.path.join(output_path, "test_fe_"+now_string+".csv"),
  }
  write_file("/tmp/train.txt", filenames['train'])
  write_file("/tmp/val.txt", filenames['val'])
  write_file("/tmp/test.txt", filenames['test'])

  return filenames

def parse_args():
  '''Parse command line arguments.'''
  parser = argparse.ArgumentParser(
      description='Feature Engineering Script.')
  parser.add_argument('--input_train_data', type=str, required=True,
      help='GCS Path to train dataset.')
  parser.add_argument('--input_val_data', type=str, required=True,
      help="GCS Path to val dataset.")
  parser.add_argument('--input_test_data', type=str, required=True,
      help="GCS Path to test dataset.")
  parser.add_argument('--output_path', type=str, required=True,
      help='GCS Path to output feature engineered datasets.')
  parser.add_argument('--log_level', type=str, default="INFO",
      help="Logging level.")
  return parser.parse_args()

def main():
  '''Entrypoint.'''
  args = parse_args()
  logging.getLogger().setLevel(args.log_level)

  train_df = pd.read_csv(args.input_train_data)
  val_df = pd.read_csv(args.input_val_data)
  test_df = pd.read_csv(args.input_test_data)

  imputer = MissingAgeImputer()
  train_df = imputer.fit_transform(train_df)
  val_df = imputer.transform(val_df)
  test_df = imputer.transform(test_df)

  feature_engineering = [create_title_feature, create_family_size,
                         create_deck_feature, fill_embarked,
                         create_ticket_freq, drop_unused_columns]

  train_df = transform_dataset(train_df, feature_engineering)
  val_df = transform_dataset(val_df, feature_engineering)
  test_df = transform_dataset(test_df, feature_engineering)

  filenames = get_output_filenames(args.output_path)
  train_df.to_csv(filenames['train'])
  val_df.to_csv(filenames['val'])
  test_df.to_csv(filenames['test'])


if __name__== "__main__":
  main()
