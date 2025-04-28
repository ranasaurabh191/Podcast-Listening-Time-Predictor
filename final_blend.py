import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train = pd.read_csv('data/train_preprocessed.csv')
test = pd.read_csv('data/test_preprocessed.csv')

# Prepare data
X = train.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# In final_blend.py, add before training
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.03, 0.05],
    'max_depth': [7, 8]
}
lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(lgbm, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best LightGBM params:", grid.best_params_)
lgbm_model = grid.best_estimator_

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31, 
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

# Validate with optimized weights
weights = [0.5, 0.3, 0.2]  # Favor Random Forest, then XGBoost, LightGBM
pred_rf = rf_model.predict(X_val)
pred_xgb = xgb_model.predict(X_val)
pred_lgbm = lgbm_model.predict(X_val)
blend_pred = weights[0] * pred_rf + weights[1] * pred_xgb + weights[2] * pred_lgbm
rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
print(f'Final Blend RMSE: {rmse}')

# Predict on test
X_test = test.drop(['id'], axis=1)
test['Listening_Time_minutes'] = (weights[0] * rf_model.predict(X_test) + 
                                 weights[1] * xgb_model.predict(X_test) + 
                                 weights[2] * lgbm_model.predict(X_test))
submission = test[['id', 'Listening_Time_minutes']]
submission.to_csv('final_blend_submission.csv', index=False)
print("Final blend submission created.")