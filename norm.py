import pandas as pd
from sklearn.preprocessing import StandardScaler

input_dir = 'data/'
output_dir = 'data/norm/'
base_file_name = 'data_banknote_authentication'
label = 'class'

train_data = pd.DataFrame(pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=','))
validation_data = pd.DataFrame(pd.read_csv(input_dir + 'validation_' + base_file_name + '.csv', sep=','))
test_data = pd.DataFrame(pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=','))

scaler = StandardScaler()
scaler.fit(train_data.drop(label, 1))


def scale(df):
    df_features = df.drop(label, 1)
    scaled_data = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled_data, columns=df_features.columns)
    scaled_df[label] = df[label]
    return scaled_df


train_data = scale(train_data)
validation_data = scale(validation_data)
test_data = scale(test_data)

train_data.to_csv(output_dir + 'train_' + base_file_name + '.csv', sep=',', index=False)
validation_data.to_csv(output_dir + 'validation_' + base_file_name + '.csv', sep=',', index=False)
test_data.to_csv(output_dir + 'test_' + base_file_name + '.csv', sep=',', index=False)