import pandas as pd
from sklearn.preprocessing import StandardScaler

input_dir = 'data/'
output_dir = 'data/norm/'
base_file_name = 'data_banknote_authentication'
label = 'class'

train_data = pd.read_csv(input_dir + 'train_' + base_file_name + '.csv', sep=',')
test_data = pd.read_csv(input_dir + 'test_' + base_file_name + '.csv', sep=',')

scaler = StandardScaler()
scaler.fit(train_data.drop(label, 1))


def print_info(df):
    print(df.drop(label, 1).describe())
    print('')


def scale(df):
    df_features = df.drop(label, 1)
    scaled_data = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled_data, columns=df_features.columns)
    scaled_df[label] = df[label]
    return scaled_df


print('Train set')
print_info(train_data)
print('Test set')
print_info(test_data)

train_data = scale(train_data)
test_data = scale(test_data)

print('')
print('Normalized')
print('')
print('Train set')
print_info(train_data)
print('Test set')
print_info(test_data)

train_data.to_csv(output_dir + 'train_' + base_file_name + '.csv', sep=',', index=False)
test_data.to_csv(output_dir + 'test_' + base_file_name + '.csv', sep=',', index=False)