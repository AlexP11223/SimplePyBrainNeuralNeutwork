import pandas as pd
from pybrain.utilities import percentError

from utils import to_xy, make_sure_path_exists
from sklearn import tree

input_dir = 'data/'
base_file_name = 'data_banknote_authentication'

attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']
label = 'class'
class_names = ['Forged', 'Genuine']

train_data = pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=',')
test_data = pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=',')

train_x, train_y = to_xy(train_data, attributes, label)
test_x, test_y = to_xy(test_data, attributes, label)

clf = tree.DecisionTreeClassifier()
clf.fit(train_x, train_y)
out = clf.predict(test_x)

print('Output for test dataset:')
print(out)

error = percentError(out, test_y)
print('Error rate: %.4f%%' % error)

# use Graphviz or for example https://dreampuf.github.io/GraphvizOnline/ to render
make_sure_path_exists('tree_output')
dot_data = tree.export_graphviz(clf, out_file="tree_output/tree.dot",
                         feature_names=attributes,
                         class_names=class_names,
                         filled=True, rounded=True,
                         special_characters=True)
