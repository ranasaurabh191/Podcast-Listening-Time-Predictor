import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Existing EDA
print(train.head())
print(train.info())
print(train.describe())
print(train.isnull().sum())

# Visualizations
# 1. Target distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['Listening_Time_minutes'], bins=50)
plt.title('Distribution of Listening Time')
plt.show()

# 2. Listening time by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train)
plt.title('Listening Time by Genre')
plt.xticks(rotation=45)
plt.show()

# 3. Listening time vs. Episode Length (numerical)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train)
plt.title('Listening Time vs. Episode Length')
plt.show()

# 4. Correlation heatmap (numerical features)
plt.figure(figsize=(10, 6))
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


train = pd.read_csv('data/train.csv')
plt.figure(figsize=(10, 6))
sns.histplot(train['Listening_Time_minutes'], bins=50)
plt.title('Distribution of Listening Time')
plt.show()
print("Skewness:", train['Listening_Time_minutes'].skew())