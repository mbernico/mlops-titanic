# Python

# ==============================================================================
# Copyright 2020 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ==============================================================================

"""RFR based imputer for Age."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

class MissingAgeImputer:
  """Imputes missing age using a Random Forest Regressor."""

  def __init__(self):
    """Constructor."""
    self.rfr = RandomForestRegressor()
    self.embarked_encoder = LabelEncoder()
    self.title_encoder = LabelEncoder()
    self.deck_encoder = LabelEncoder()
    self.sex_encoder = LabelEncoder()

  def _label_encode_data(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
    """Label encodes a dataframe"""
    if fit:
      self.embarked_encoder.fit(df['Embarked'])
      self.sex_encoder.fit(df['Sex'])

    df.loc[:,'Embarked'] = self.embarked_encoder.transform(df['Embarked'])
    df.loc[:,'Sex'] = self.sex_encoder.transform(df['Sex'])
    return df

  def fit(self, df: pd.DataFrame):
    """Trains a Random Forest Regressor on the Train dataset."""
    data = df[['Pclass', 'Sex', 'Age','SibSp', 'Parch', 'Fare', 'Embarked']]
    data = data.dropna()
    data = self._label_encode_data(data, fit=True)
    X = data.drop(columns=['Age'])
    y = data[['Age']]
    self.rfr.fit(X,np.ravel(y))

  def transform(self, df:pd.DataFrame) -> pd.DataFrame:
    """Imputes Age."""
    impute_df = df[
        ['Pclass', 'Sex', 'Age','SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    missing_idx = impute_df[impute_df['Age'].isnull()].index
    if not missing_idx.empty:
      impute_df = impute_df.loc[missing_idx]
      impute_df = self._label_encode_data(impute_df)
      impute_df = impute_df.drop(columns=['Age'])
      imputed_ages = self.rfr.predict(impute_df)
      df.loc[missing_idx,'Age'] = np.round(imputed_ages)
    return df

  def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Fits the Random Forest Regressor imputer on the train dataset
      and imputes missing values for Age."""
    self.fit(df)
    df = self.transform(df)
    return df
