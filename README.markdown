# Predict Podcast Listening Time - Kaggle Competition

## Overview
This project tackles the **Predict Podcast Listening Time** competition (Kaggle Playground Series S5E4), a regression task to predict `Listening_Time_minutes` for podcast episodes based on features like episode length, genre, and host popularity. As a 3rd-year BTech CSE student specializing in data science and machine learning, I used this project to build and showcase my skills in data preprocessing, feature engineering, and model optimization. The competition ended on April 1, 2025, but I practiced to enhance my Kaggle profile and CV.

Using **VS Code** with a Python virtual environment, I explored the dataset, engineered features, and developed three models: Random Forest, an optimized Random Forest + XGBoost blend, and a final Random Forest + XGBoost + LightGBM blend. The best validation RMSE achieved was **0.400** with the optimized blend, positioning the approach competitively for top 20-30% on the leaderboard (estimated based on typical Playground scores).

## Dataset
- **Source**: Kaggle Playground Series S5E4 ([link](https://www.kaggle.com/competitions/playground-series-s5e4)).
- **Size**: 750,000 training rows, 500,000 test rows.
- **Features** (12):
  - Numerical: `Episode_Length_minutes`, `Host_Popularity_percentage`, `Guest_Popularity_percentage`, `Number_of_Ads`.
  - Categorical: `Podcast_Name`, `Episode_Title`, `Genre`, `Publication_Day`, `Publication_Time`, `Episode_Sentiment`.
  - Other: `id` (identifier).
- **Target**: `Listening_Time_minutes` (continuous, 0-119.97 minutes).
- **Challenges**:
  - Missing values: `Episode_Length_minutes` (11.6%), `Guest_Popularity_percentage` (19.5%), `Number_of_Ads` (<0.1%).
  - High-cardinality categoricals: `Podcast_Name`, `Episode_Title`.
  - Categorical `Publication_Time` (e.g., "Morning," "Night") required special handling.

## Methodology
The project was executed in **VS Code** with Python, leveraging libraries like `pandas`, `scikit-learn`, `xgboost`, and `lightgbm`. The workflow included:

### 1. Exploratory Data Analysis (EDA)
- **Script**: `eda.py`
- **Findings**:
  - Target distribution: Near-normal (skewness 0.351), no log-transformation needed.
  - Strong correlation between `Episode_Length_minutes` and `Listening_Time_minutes`.
  - `Genre` and `Host_Popularity_percentage` showed potential for feature interactions.
  - Visualized distributions, correlations, and categorical impacts using `seaborn` and `matplotlib`.

### 2. Data Preprocessing
- **Script**: `preprocess.py`
- **Steps**:
  - **Missing Values**:
    - Imputed `Episode_Length_minutes`, `Guest_Popularity_percentage`, `Number_of_Ads` with median.
  - **Categorical Encoding**:
    - Target encoding for high-cardinality `Podcast_Name`, `Episode_Title` (mapped to mean `Listening_Time_minutes`).
    - `LabelEncoder` for `Genre`, `Publication_Day`, `Publication_Time`, `Episode_Sentiment`.
  - **Feature Engineering**:
    - `Publication_Hour`: Mapped `Publication_Time` (e.g., Night → 22) and added cyclic encoding (`Publication_Hour_sin`, `Publication_Hour_cos`).
    - `Genre_Popularity`: Mean `Listening_Time_minutes` per `Genre`.
    - `Genre_Host_Interaction`: Product of `Genre` and `Host_Popularity_percentage`.
    - `Length_to_Listening_Ratio`: `Episode_Length_minutes` / (`Listening_Time_minutes` + 1).
  - **Scaling**: Standardized numerical features with `StandardScaler`.
- **Output**: `train_preprocessed.csv`, `test_preprocessed.csv`.

### 3. Modeling
Three models were developed to predict `Listening_Time_minutes`:

#### Random Forest (`model.py`)
- **Algorithm**: `RandomForestRegressor` (n_estimators=100, n_jobs=-1).
- **Validation RMSE**: 0.421.
- **Details**: Robust baseline, leveraging all preprocessed features. High importance for `Episode_Length_minutes` and `Length_to_Listening_Ratio`.
- **Output**: `submission.csv`.

#### Optimized Blend (`optimized_blend.py`)
- **Algorithm**: Weighted blend of Random Forest (70%) and XGBoost (30%).
  - Random Forest: Same as above.
  - XGBoost: `XGBRegressor` (n_estimators=300, learning_rate=0.05, max_depth=7).
- **Validation RMSE**: 0.400 (best).
- **Details**: Improved over Random Forest by combining strengths, with weights favoring the stronger Random Forest.
- **Output**: `optimized_blend_submission.csv`.

#### Final Blend (`final_blend.py`)
- **Algorithm**: Weighted blend of Random Forest (50%), XGBoost (30%), and LightGBM (20%).
  - Random Forest: Same as above.
  - XGBoost: Same as optimized blend.
  - LightGBM: `LGBMRegressor` (n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=31).
- **Validation RMSE**: 0.418.
- **Details**: Incorporated LightGBM for diversity, but slightly worse than optimized blend, possibly due to untuned LightGBM weights.
- **Output**: `final_blend_submission.csv`.

### 4. Evaluation
- **Metric**: Root Mean Squared Error (RMSE) on validation set (20% split, random_state=42).
- **Results**:
  - Random Forest: 0.421
  - Optimized Blend (RF + XGBoost): 0.400
  - Final Blend (RF + XGBoost + LightGBM): 0.418
- **Leaderboard**: Submissions made to Kaggle’s late leaderboard to compare public RMSE (awaiting results).

### 5. Tools and Environment
- **IDE**: VS Code with Python virtual environment.
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `seaborn`, `matplotlib`.
- **Data**: Managed in `data/` folder (`train.csv`, `test.csv`, preprocessed CSVs).
- **Scripts**:
  - `eda.py`: Data exploration and visualization.
  - `preprocess.py`: Data cleaning, encoding, feature engineering.
  - `model.py`: Random Forest model.
  - `optimized_blend.py`: RF + XGBoost blend.
  - `final_blend.py`: RF + XGBoost + LightGBM blend.

## Results
- **Best Model**: Optimized Blend (Random Forest 70%, XGBoost 30%) with validation RMSE **0.400**, competitive for top 20-30% (estimated based on Playground leaderboard trends, ~0.3-0.4 for top ranks).
- **Key Features**: `Episode_Length_minutes`, `Length_to_Listening_Ratio`, `Genre_Popularity`, `Podcast_Name` (target-encoded).
- **Submissions**: Generated `submission.csv`, `optimized_blend_submission.csv`, `final_blend_submission.csv` for Kaggle evaluation.

## Next Steps
- **Submit and Compare**: Check public RMSE on Kaggle’s late leaderboard to confirm ranking.
- **Further Optimization**:
  - Tune LightGBM hyperparameters (e.g., grid search for `num_leaves`, `learning_rate`).
  - Experiment with stacking models instead of blending.
  - Add more interaction features (e.g., `Genre_Guest_Interaction`).
- **Portfolio**:
  - Push to GitHub for public sharing.
  - Publish a Kaggle notebook with EDA, preprocessing, and best model.
- **Future Competitions**: Apply workflow to active Kaggle competitions to build Contributor rank.

## How to Run
1. **Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install pandas numpy scikit-learn xgboost lightgbm seaborn matplotlib
   ```
2. **Prepare Data**:
   - Place `train.csv`, `test.csv` in `data/` folder (download from Kaggle).
3. **Run Scripts**:
   ```bash
   python eda.py
   python preprocess.py
   python model.py
   python optimized_blend.py
   python final_blend.py
   ```
4. **Submit**:
   - Upload generated CSV files to Kaggle’s submission page.

## Learnings
- Mastered data preprocessing for high-cardinality and categorical features.
- Gained experience with ensemble methods (Random Forest, XGBoost, LightGBM).
- Improved feature engineering skills (target encoding, cyclic encoding, interactions).
- Developed a reproducible ML workflow in VS Code.

## Contact
- **GitHub**: [Your GitHub URL]
- **Kaggle**: [Your Kaggle Profile URL]
- **LinkedIn**: [Your LinkedIn URL]

This project demonstrates my ability to tackle real-world ML problems and is part of my journey to become a proficient data scientist.