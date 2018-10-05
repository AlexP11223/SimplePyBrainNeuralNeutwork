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
