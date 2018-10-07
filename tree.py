import pandas as pd
from pybrain.utilities import percentError
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from utils import to_xy, make_sure_path_exists
from sklearn import tree

input_dir = 'data/'
base_file_name = 'data_banknote_authentication'

attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']
label = 'class'
class_names = ['Forged', 'Genuine']

max_depth = None
min_samples_split = 2
min_samples_leaf = 2

train_data = pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=',')
test_data = pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=',')

train_x, train_y = to_xy(train_data, attributes, label)
test_x, test_y = to_xy(test_data, attributes, label)

clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)
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

# try different max_depths to check overfitting

max_depths = np.linspace(1, 10, 10, endpoint=True)

train_results = []
test_results = []
for val in max_depths:
    dt = tree.DecisionTreeClassifier(max_depth=val)
    dt.fit(train_x, train_y)

    train_out = dt.predict(train_x)
    test_out = dt.predict(test_x)

    train_results.append(percentError(train_out, train_y) / 100)
    test_results.append(percentError(test_out, test_y) / 100)

line1, = plt.plot(max_depths, train_results, 'b', label='Train error')
line2, = plt.plot(max_depths, test_results, 'r', label='Test error')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Error')
plt.xlabel('Max tree depth')
plt.show()

# try different min_samples_splits to check underfitting

min_samples_splits = np.linspace(2, 10, 10, endpoint=True)

train_results = []
test_results = []
for val in min_samples_splits:
    dt = tree.DecisionTreeClassifier(min_samples_split=int(val))
    dt.fit(train_x, train_y)

    train_out = dt.predict(train_x)
    test_out = dt.predict(test_x)

    train_results.append(percentError(train_out, train_y) / 100)
    test_results.append(percentError(test_out, test_y) / 100)

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train error')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test error')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Error')
plt.xlabel('Min samples split')
plt.show()

# try different min_samples_leafs to check underfitting

min_samples_leafs = np.linspace(1, 10, 10, endpoint=True)

train_results = []
test_results = []
for val in min_samples_leafs:
    dt = tree.DecisionTreeClassifier(min_samples_leaf=int(val))
    dt.fit(train_x, train_y)

    train_out = dt.predict(train_x)
    test_out = dt.predict(test_x)

    train_results.append(percentError(train_out, train_y) / 100)
    test_results.append(percentError(test_out, test_y) / 100)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train error')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test error')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Error')
plt.xlabel('Min samples leaf')
plt.show()