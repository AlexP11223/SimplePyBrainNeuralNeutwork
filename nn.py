import pandas as pd
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from matplotlib import pyplot

from nnvisualizer import PybrainNNVisualizer

input_dir = 'data/norm/'
base_file_name = 'data_banknote_authentication'

attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']
label = 'class'
class_names = ['Forged', 'Genuine']

hidden_neurons_count = 8

def to_dataset(df):
    ds = ClassificationDataSet(len(attributes), 1, class_labels=class_names)
    for i, row in df.iterrows():
        input_row = [row[attr] for attr in attributes]
        output_row = (row[label])
        ds.addSample(input_row, output_row)
    return ds


train_data = pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=',')
test_data = pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=',')

train_ds = to_dataset(train_data)
test_ds = to_dataset(test_data)

fnn = buildNetwork(train_ds.indim, hidden_neurons_count, train_ds.outdim)

print(fnn)
PybrainNNVisualizer(fnn).draw()

print('Training')

trainer = BackpropTrainer(fnn, train_ds, verbose=True)
train_out = trainer.trainUntilConvergence(maxEpochs=100, validationProportion=0.5)

pyplot.plot(train_out[0], label='Training set')
pyplot.plot(train_out[1], label='Validation set')
pyplot.xlabel('Epochs')
pyplot.ylabel('Error')
pyplot.legend()
pyplot.show()

out = fnn.activateOnDataset(test_ds)

print('Output for test dataset:')
out_values = [it[0] for it in out.tolist()]
print(out_values)
out_values = [int(abs(round(it))) for it in out_values]
print(out_values)
