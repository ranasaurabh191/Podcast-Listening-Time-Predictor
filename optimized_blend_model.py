import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train = pd.read_csv('data/train_preprocessed.csv')
test = pd.read_csv('data/test_preprocessed.csv')

# Prepare data
X = train.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Validate with different weights
weights = [0.7, 0.3]  # Favor Random Forest
pred_rf = rf_model.predict(X_val)
pred_xgb = xgb_model.predict(X_val)
blend_pred = weights[0] * pred_rf + weights[1] * pred_xgb
rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
print(f'Optimized Blend RMSE: {rmse}')

# Predict on test
X_test = test.drop(['id'], axis=1)
test['Listening_Time_minutes'] = weights[0] * rf_model.predict(X_test) + weights[1] * xgb_model.predict(X_test)
submission = test[['id', 'Listening_Time_minutes']]
submission.to_csv('optimized_blend_submission.csv', index=False)
print("Optimized blend submission created.")