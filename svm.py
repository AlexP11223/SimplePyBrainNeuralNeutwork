import pandas as pd
from utils import to_xy
from sklearn import svm
from pybrain.utilities import percentError


input_dir = 'data/norm/'
base_file_name = 'data_banknote_authentication'

attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']
label = 'class'
class_names = ['Forged', 'Genuine']

train_data = pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=',')
test_data = pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=',')

train_x, train_y = to_xy(train_data, attributes, label)
test_x, test_y = to_xy(test_data, attributes, label)

clf = svm.SVC()
clf.fit(train_x, train_y)
out = clf.predict(test_x)

print('Output for test dataset:')
print(out)

error = percentError(out, test_y)
print('Error rate: %.4f%%' % error)
