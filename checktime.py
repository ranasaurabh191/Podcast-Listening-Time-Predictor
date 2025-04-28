import pandas as pd

# Load data
train = pd.read_csv('data/train.csv')
print("Unique values in Publication_Time:")
print(train['Publication_Time'].unique())
print("Value counts:")
print(train['Publication_Time'].value_counts())