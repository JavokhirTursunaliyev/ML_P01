import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("COMPLETE ML PIPELINE: FEATURE ENGINEERING → MODEL TRAINING → EVALUATION")
print("="*100)

# ============================================================================
# STEP 1: LOAD & MERGE DATA
# ============================================================================
print("\n[STEP 1] Loading and merging datasets...")
df_parcel = pd.read_csv('EXTR_Parcel.csv', encoding='latin-1')
df_sales = pd.read_csv('EXTR_RPSale.csv', encoding='latin-1')
print(f"✓ Parcel: {df_parcel.shape} | Sales: {df_sales.shape}")

df_merged = pd.merge(df_parcel, df_sales, on=['Major', 'Minor'], how='inner')
print(f"✓ Merged: {df_merged.shape}")

# Identify target
price_cols = [col for col in df_merged.columns if 'Price' in col]
target_col = price_cols[0] if price_cols else None
print(f"✓ Target variable: {target_col}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING  
# ============================================================================
print("\n[STEP 2] Feature engineering...")

# Create composite features
view_features = ['MtRainier', 'Olympics', 'Cascades', 'SeattleSkyline', 'PugetSound', 
                  'LakeWashington', 'LakeSammamish', 'SmallLakeRiverCreek', 'OtherView']
df_merged['ViewScore'] = df_merged[[f for f in view_features if f in df_merged.columns]].sum(axis=1)

accessibility_features = ['WaterSystem', 'SewerSystem', 'Access']
df_merged['AccessibilityScore'] = df_merged[[f for f in accessibility_features if f in df_merged.columns]].mean(axis=1)

# Area interaction
df_merged['AreaInteraction'] = df_merged['Area'] * df_merged['SubArea']

# Hazard/nuisance indicator
hazard_features = [col for col in df_merged.columns if any(x in col for x in ['Hazard', 'Problem', 'Noise'])]
df_merged['HazardCount'] = (df_merged[[f for f in hazard_features if f in df_merged.columns]] > 0).sum(axis=1)

print(f"✓ Created 4 composite features: ViewScore, AccessibilityScore, AreaInteraction, HazardCount")

# ============================================================================
# STEP 3: DATA CLEANING & SCALING
# ============================================================================
print("\n[STEP 3] Cleaning and scaling...")

# Remove rows with missing target
df_ml = df_merged.dropna(subset=[target_col]).copy()
print(f"✓ Removed {len(df_merged) - len(df_ml)} rows with missing target")

# Select features
numeric_features = df_ml.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in ['Major', 'Minor', target_col]]

# Scale numeric features
scaler = RobustScaler()
df_ml[numeric_features] = scaler.fit_transform(df_ml[numeric_features])

print(f"✓ Scaled {len(numeric_features)} numeric features")

# ============================================================================
# STEP 4: EXPLORATORY ANALYSIS
# ============================================================================
print("\n[STEP 4] Exploratory analysis...")

# Feature correlations
corr_with_target = df_ml[numeric_features + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
print(f"\n✓ Top 10 correlations with {target_col}:")
for i, (feat, corr_val) in enumerate(corr_with_target.head(10).items(), 1):
    print(f"  {i:2d}. {feat:50s}: {corr_val:7.4f}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 5] Creating train/test split...")

X = df_ml[numeric_features]
y = df_ml[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================================
# STEP 6: BASELINE MODELS
# ============================================================================
print("\n[STEP 6] Training baseline models...")
models = {}

# Model 1: Linear Regression
print("\n  Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
models['Linear Regression'] = {
    'model': lr,
    'pred': y_pred_lr,
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'mae': mean_absolute_error(y_test, y_pred_lr),
    'r2': r2_score(y_test, y_pred_lr)
}

# Model 2: Random Forest
print("  Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
models['Random Forest'] = {
    'model': rf,
    'pred': y_pred_rf,
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2': r2_score(y_test, y_pred_rf)
}

# Model 3: Gradient Boosting
print("  Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
models['Gradient Boosting'] = {
    'model': gb,
    'pred': y_pred_gb,
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'mae': mean_absolute_error(y_test, y_pred_gb),
    'r2': r2_score(y_test, y_pred_gb)
}

# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================
print("\n[STEP 7] Model performance comparison:")
print("-"*100)
print(f"{'Model':<30} {'RMSE':>15} {'MAE':>15} {'R²':>15}")
print("-"*100)

best_model = None
best_r2 = -np.inf

for model_name, metrics in sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"{model_name:<30} ${metrics['rmse']:>14,.0f} ${metrics['mae']:>14,.0f} {metrics['r2']:>15.4f}")
    if metrics['r2'] > best_r2:
        best_r2 = metrics['r2']
        best_model = model_name

print("-"*100)
print(f"✓ Best Model: {best_model} (R² = {best_r2:.4f})")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
print(f"\n[STEP 8] Top 15 features ({best_model}):")
print("-"*100)

if best_model == 'Random Forest':
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
elif best_model == 'Gradient Boosting':
    importances = pd.Series(gb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
else:
    importances = pd.Series(np.abs(lr.coef_), index=X_train.columns).sort_values(ascending=False)

for i, (feat, imp) in enumerate(importances.head(15).items(), 1):
    print(f"  {i:2d}. {feat:50s}: {imp:10.6f}")

print("\n" + "="*100)
print("PIPELINE COMPLETE!")
print("="*100)

# Save results
results_summary = f"""
ML PIPELINE RESULTS SUMMARY
{'='*60}

DATASET:
  Merged Records: {len(df_ml):,}
  Features: {len(numeric_features)}
  Target Variable: {target_col}

MODELS TRAINED:
{chr(10).join([f"  • {name}: R²={metrics['r2']:.4f}, RMSE=${metrics['rmse']:,.0f}" for name, metrics in models.items()])}

BEST MODEL: {best_model}
  • R² Score: {best_r2:.4f}
  • RMSE: ${models[best_model]['rmse']:,.0f}
  • MAE: ${models[best_model]['mae']:,.0f}

TOP 5 FEATURES:
{chr(10).join([f"  {i}. {feat}" for i, feat in enumerate(importances.head(5).index, 1)])}
"""

with open('ml_results.txt', 'w') as f:
    f.write(results_summary)

print("\n✓ Results saved to 'ml_results.txt'")
