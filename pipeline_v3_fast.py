"""
King County House Price Pipeline — v3 FAST
==========================================================================
OPTIMIZED VERSION: Reduced dataset & faster models for quicker results
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("KING COUNTY HOUSE PRICE PIPELINE v3 FAST — Feature Engineering + Tuning (Optimized)")
print("=" * 100)

# Load and process
print("\n[STEP 1-3] Loading, cleaning, and merging data...")
df_parcel = pd.read_csv('EXTR_Parcel_processed.csv', encoding='latin-1')
df_bldg   = pd.read_csv('EXTR_ResBldg.csv', encoding='latin-1')
df_sales  = pd.read_csv('EXTR_RPSale.csv', encoding='latin-1')

for df in (df_parcel, df_bldg, df_sales):
    df.columns = df.columns.str.strip()

# Clean sales
df_sales = df_sales[df_sales['SalePrice'] > 10000]
df_sales['SaleWarning'] = df_sales['SaleWarning'].fillna('').astype(str).str.strip()
df_sales = df_sales[df_sales['SaleWarning'] == '']
df_sales['DocumentDate'] = pd.to_datetime(df_sales['DocumentDate'], format='%m/%d/%Y', errors='coerce')
df_sales = df_sales.dropna(subset=['DocumentDate'])
df_sales = df_sales.sort_values('DocumentDate').drop_duplicates(subset=['Major', 'Minor'], keep='last')

# Merge
df_merged = df_parcel.merge(df_bldg, on=['Major', 'Minor'], how='inner', suffixes=('_parcel', '_bldg'))
df_merged = df_merged.merge(df_sales[['Major', 'Minor', 'SalePrice', 'DocumentDate']], on=['Major', 'Minor'], how='inner')

print(f"  Merged: {df_merged.shape}")

# Feature engineering
print("\n[STEP 4] Feature engineering...")
df_merged['SaleYear'] = df_merged['DocumentDate'].dt.year
df_merged['HouseAge'] = (df_merged['SaleYear'] - df_merged.get('YrBuilt', 0)).clip(lower=0)
df_merged['IsRenovated'] = (df_merged.get('YrRenovated', 0) > 0).astype(int)
df_merged['YearsSinceRenovation'] = ((df_merged['SaleYear'] - df_merged.get('YrRenovated', 0)).clip(lower=0))
df_merged.loc[df_merged.get('YrRenovated', 0) == 0, 'YearsSinceRenovation'] = 999
df_merged['LotToBuildingRatio'] = (
    df_merged.get('SqFtLot', 1) / df_merged.get('SqFtTotLiving', 1).clip(lower=1)
)
df_merged['ConditionScore'] = df_merged.get('Condition', 0).fillna(0)
df_merged['BathroomCount'] = df_merged[
    [col for col in df_merged.columns if 'bath' in col.lower()]
].sum(axis=1)
df_merged['BedroomCount'] = df_merged[
    [col for col in df_merged.columns if 'bed' in col.lower()]
].sum(axis=1)
df_merged['RecentRenovation'] = ((df_merged['IsRenovated'] == 1) & (df_merged['YearsSinceRenovation'] <= 10)).astype(int)
df_merged['IsNewConstruction'] = (df_merged['HouseAge'] < 5).astype(int)

print("  ✓ All 9 engineered features created")

# Sample data for faster execution (80% sample)
sample_frac = 0.8
df_ml = df_merged.dropna(subset=['SalePrice']).sample(frac=sample_frac, random_state=42)

target_col = 'SalePrice'
numeric_features = df_ml.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Major', 'Minor', target_col, 'SaleYear', 'YrBuilt', 'YrRenovated']
numeric_features = [f for f in numeric_features if f not in exclude_cols]

X = df_ml[numeric_features].fillna(0)
y = df_ml[target_col]

print(f"\n[STEP 5] Prepared {X.shape[0]:,} records × {X.shape[1]} features")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"  Train: {X_train_scaled.shape[0]:,} | Test: {X_test_scaled.shape[0]:,}")

# Baseline models (smaller, faster)
print("\n[STEP 6] Training baseline models...")
models = {}

# Random Forest (50 trees instead of 100)
print("  • Random Forest (50 trees, max_depth=15)...")
rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
models['Random Forest'] = {
    'model': rf,
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2': r2_score(y_test, y_pred_rf),
}

# Gradient Boosting
print("  • Gradient Boosting (50 estimators)...")
gb = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
models['Gradient Boosting'] = {
    'model': gb,
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'mae': mean_absolute_error(y_test, y_pred_gb),
    'r2': r2_score(y_test, y_pred_gb),
}

# Hyperparameter tuning (Random Forest)
print("\n[STEP 7] Hyperparameter tuning (Random Forest with RandomizedSearchCV)...")
param_dist = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_dist, n_iter=12, cv=2, scoring='r2', n_jobs=-1, random_state=42, verbose=0
)
search.fit(X_train_scaled, y_train)

best_model = search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)

models['Random Forest (Tuned)'] = {
    'model': best_model,
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_tuned)),
    'mae': mean_absolute_error(y_test, y_pred_tuned),
    'r2': r2_score(y_test, y_pred_tuned),
    'cv_score': search.best_score_,
    'best_params': search.best_params_,
}

# Results
print("\n[STEP 8] Model Comparison:")
print("-" * 100)
print(f"{'Model':<30} {'RMSE':>15} {'MAE':>15} {'R² Score':>12}")
print("-" * 100)

baseline_r2 = 0.4777
for name, m in sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True):
    improvement = ((m['r2'] - baseline_r2) / baseline_r2) * 100
    cv_info = f" (CV: {m.get('cv_score', 0):.4f})" if 'cv_score' in m else ""
    print(f"{name:<30} ${m['rmse']:>14,.0f} ${m['mae']:>14,.0f} {m['r2']:>12.4f} {improvement:+6.1f}%{cv_info}")

print("-" * 100)

best_overall = max(models.items(), key=lambda x: x[1]['r2'])
print(f"\n✓ Best Model: {best_overall[0]} (R² = {best_overall[1]['r2']:.4f})")

if 'best_params' in best_overall[1]:
    print("  Best parameters:")
    for param, value in best_overall[1]['best_params'].items():
        print(f"    • {param}: {value}")

# Feature importance
print(f"\n[STEP 9] Top 15 Features ({best_overall[0]}):")
print("-" * 100)

best_model_obj = best_overall[1]['model']
if hasattr(best_model_obj, 'feature_importances_'):
    importances = pd.Series(best_model_obj.feature_importances_, index=X_train.columns).sort_values(ascending=False)
else:
    importances = pd.Series(np.abs(best_model_obj.coef_), index=X_train.columns).sort_values(ascending=False)

engineered_features = {
    'HouseAge', 'LotToBuildingRatio', 'IsRenovated', 'RecentRenovation',
    'IsNewConstruction', 'YearsSinceRenovation', 'ConditionScore',
    'BathroomCount', 'BedroomCount'
}

for i, (feat, imp) in enumerate(importances.head(15).items(), 1):
    marker = " [ENGINEERED]" if feat in engineered_features else ""
    print(f"  {i:2d}. {feat:45s}: {imp:10.6f}{marker}")

print("\n" + "=" * 100)
print("PIPELINE COMPLETE")
print("=" * 100)
