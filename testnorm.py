import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest

file_path = 'data/train_data_banknote_authentication.csv'
attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']

df = pd.DataFrame(pd.read_csv(file_path, sep=','))

for attr in attributes:
    print(attr)
    stat, p = normaltest(df[attr])
    print('p=%.3f' % p)
    pyplot.hist(df[attr])
    pyplot.title(attr)
    pyplot.show()
    qqplot(df[attr], line='s')
    pyplot.title(attr)
    pyplot.show()
