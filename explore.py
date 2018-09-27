import pandas as pd
from matplotlib import pyplot
import seaborn as sns

file_path = 'data/data_banknote_authentication.csv'
attributes = ['varianceWT', 'skewnessWT', 'curtosisWT', 'entropy']
label = 'class'

df = pd.read_csv(file_path, sep=',')

print(df.describe())

sns.set_palette(sns.xkcd_palette(['pale red', 'medium green']))

sns.countplot(x=label, data=df)
pyplot.show()

sns.pairplot(data=df, hue=label)
pyplot.show()