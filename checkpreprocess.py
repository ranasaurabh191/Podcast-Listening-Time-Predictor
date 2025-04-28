# check_preprocessed.py
import pandas as pd
train = pd.read_csv('data/train_preprocessed.csv')
test = pd.read_csv('data/test_preprocessed.csv')
print("Train head:")
print(train.head())
print("Train info:")
print(train.info())
print("Test head:")
print(test.head())
print("Test info:")
print(test.info())