import errno
import os

import pandas as pd
from pybrain.datasets import ClassificationDataSet


def to_dataset(df, attributes, class_names, label):
    # type: (pd.DataFrame, list, list, str) -> ClassificationDataSet
    ds = ClassificationDataSet(len(attributes), 1, class_labels=class_names)
    for i, row in df.iterrows():
        input_row = [row[attr] for attr in attributes]
        output_row = (row[label])
        ds.addSample(input_row, output_row)
    return ds


def to_xy(df, attributes, label):
    # type: (pd.DataFrame, list, str) -> [pd.DataFrame, pd.DataFrame]
    return df[attributes], df[label]


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
