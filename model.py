import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load preprocessed data
train = pd.read_csv('data/train_preprocessed.csv')

# Prepare features and target
X = train.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse}')

# Save predictions for test set
test = pd.read_csv('data/test_preprocessed.csv')
X_test = test.drop(['id'], axis=1)
test['Listening_Time_minutes'] = model.predict(X_test)
submission = test[['id', 'Listening_Time_minutes']]
submission.to_csv('submission.csv', index=False)
print("Submission file created.")