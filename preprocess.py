import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Handle missing values
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median())
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].median())
test['Number_of_Ads'] = test['Number_of_Ads'].fillna(train['Number_of_Ads'].median())

# Target encoding for high-cardinality features
for col in ['Podcast_Name', 'Episode_Title']:
    mean_target = train.groupby(col)['Listening_Time_minutes'].mean()
    train[col] = train[col].map(mean_target)
    test[col] = test[col].map(mean_target).fillna(mean_target.mean())

# Feature engineering: Map Publication_Time
time_mapping = {'Morning': 8, 'Afternoon': 14, 'Evening': 18, 'Night': 22}
train['Publication_Hour'] = train['Publication_Time'].map(time_mapping).fillna(12)
test['Publication_Hour'] = test['Publication_Time'].map(time_mapping).fillna(12)

# Additional feature: Genre popularity
genre_popularity = train.groupby('Genre')['Listening_Time_minutes'].mean()
train['Genre_Popularity'] = train['Genre'].map(genre_popularity)
test['Genre_Popularity'] = test['Genre'].map(genre_popularity).fillna(genre_popularity.mean())

# Encode categorical features
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
le = LabelEncoder()
for col in categorical_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Feature engineering: Length ratio
train['Length_to_Listening_Ratio'] = train['Episode_Length_minutes'] / (train['Listening_Time_minutes'] + 1)
test['Length_to_Listening_Ratio'] = test['Episode_Length_minutes'] / (test['Episode_Length_minutes'].mean() + 1)

# Scale numerical features
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Hour', 
                 'Length_to_Listening_Ratio', 'Podcast_Name', 'Episode_Title', 'Genre_Popularity']
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# Save
train.to_csv('data/train_preprocessed.csv', index=False)
test.to_csv('data/test_preprocessed.csv', index=False)
print("Preprocessed data saved.")