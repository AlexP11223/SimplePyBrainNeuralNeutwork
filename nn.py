import pandas as pd
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from matplotlib import pyplot
from pybrain.utilities import percentError
from nnvisualizer import PybrainNNVisualizer
from utils import to_dataset

input_dir = 'data/norm/'
base_file_name = 'data_banknote_authentication'

attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']
label = 'class'
class_names = ['Forged', 'Genuine']

hidden_neurons_count = 8
max_epoch = 100
validation_proportion = 0.5
learning_rate = 0.1

train_data = pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=',')
test_data = pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=',')

train_ds = to_dataset(train_data, attributes, class_names, label)
test_ds = to_dataset(test_data, attributes, class_names, label)

fnn = buildNetwork(train_ds.indim, hidden_neurons_count, train_ds.outdim)

print(fnn)
PybrainNNVisualizer(fnn).draw()

print('Training')

trainer = BackpropTrainer(fnn, train_ds, learningrate=learning_rate, verbose=True)
train_out = trainer.trainUntilConvergence(maxEpochs=max_epoch, validationProportion=validation_proportion)

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

error = percentError(out_values, test_ds.getField('target'))
print('Error rate: %.4f%%' % error)
