"""
King County House Price Pipeline — v2
Fixes vs. analyze.py:
  1. Actually reads EXTR_RPSale.csv (was missing entirely)
  2. Merges Parcel + ResBldg + RPSale (was only Parcel + Sales — no building features)
  3. Filters non-arms-length sales using SaleWarning (family transfers, etc. aren't market prices)
  4. Deduplicates to most-recent sale per parcel (parcels have ~2.2 sales on avg — raw join
     would duplicate parcel/building rows across every historical sale)
  5. Fits scaler on TRAIN ONLY, transforms test — analyze.py scaled before splitting,
     which leaks test-set distribution info into training
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("KING COUNTY HOUSE PRICE PIPELINE v2")
print("=" * 100)

# ============================================================================
# STEP 1: LOAD
# ============================================================================
print("\n[STEP 1] Loading raw extracts...")
df_parcel = pd.read_csv('EXTR_Parcel_processed.csv', encoding='latin-1')          # already one-hot encoded
df_bldg   = pd.read_csv('EXTR_ResBldg.csv', encoding='latin-1')
df_sales  = pd.read_csv('EXTR_RPSale.csv', encoding='latin-1')

for name, df in [('Parcel', df_parcel), ('ResBldg', df_bldg), ('RPSale', df_sales)]:
    print(f"  {name:10s}: {df.shape[0]:>9,} rows x {df.shape[1]:>3} cols")

# Column names sometimes carry stray whitespace/quotes from the raw export
for df in (df_parcel, df_bldg, df_sales):
    df.columns = df.columns.str.strip()

# ============================================================================
# STEP 2: CLEAN SALES — filter to real, arms-length, residential sales
# ============================================================================
print("\n[STEP 2] Cleaning sales records...")
print(f"  Raw sale records: {len(df_sales):,}")

# Drop $0 / near-zero prices (gifts, corrections, family transfers priced at $1)
df_sales = df_sales[df_sales['SalePrice'] > 10000]
print(f"  After SalePrice > $10,000 filter: {len(df_sales):,}")

# SaleWarning holds space-separated 2-char codes flagging non-arms-length sales.
# Any non-null/non-empty warning code disqualifies the sale as a market transaction.
df_sales['SaleWarning'] = df_sales['SaleWarning'].fillna('').astype(str).str.strip()
df_sales = df_sales[df_sales['SaleWarning'] == '']
print(f"  After removing SaleWarning-flagged sales: {len(df_sales):,}")

# Parse date, keep most recent sale per parcel (Major+Minor can have many historical sales)
df_sales['DocumentDate'] = pd.to_datetime(df_sales['DocumentDate'], format='%m/%d/%Y', errors='coerce')
df_sales = df_sales.dropna(subset=['DocumentDate'])
df_sales = df_sales.sort_values('DocumentDate').drop_duplicates(subset=['Major', 'Minor'], keep='last')
print(f"  After keeping most-recent sale per parcel: {len(df_sales):,}")

# ============================================================================
# STEP 3: MERGE — Parcel + ResBldg + cleaned Sales
# ============================================================================
print("\n[STEP 3] Merging parcel + building + sales on Major/Minor...")
df_merged = df_parcel.merge(df_bldg, on=['Major', 'Minor'], how='inner', suffixes=('', '_bldg'))
df_merged = df_merged.merge(df_sales[['Major', 'Minor', 'SalePrice', 'DocumentDate']],
                             on=['Major', 'Minor'], how='inner')
print(f"  Merged (has parcel + building + valid sale): {df_merged.shape}")

target_col = 'SalePrice'

# ============================================================================
# STEP 4: FEATURE PREP
# ============================================================================
print("\n[STEP 4] Preparing features...")

numeric_features = df_merged.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in ['Major', 'Minor', target_col]]

df_ml = df_merged.dropna(subset=[target_col]).copy()
print(f"  Rows with valid target: {len(df_ml):,} | Candidate numeric features: {len(numeric_features)}")

X = df_ml[numeric_features].fillna(0)
y = df_ml[target_col]

# ============================================================================
# STEP 5: SPLIT FIRST, THEN SCALE (fit on train only — no leakage)
# ============================================================================
print("\n[STEP 5] Train/test split, then scaling...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
print(f"  Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")

# ============================================================================
# STEP 6: TRAIN MODELS
# ============================================================================
print("\n[STEP 6] Training models...")
models = {}

for name, model in [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
]:
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    models[name] = {
        'model': model,
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred),
        'r2': r2_score(y_test, pred),
    }

# ============================================================================
# STEP 7: COMPARE
# ============================================================================
print("\n[STEP 7] Model comparison:")
print("-" * 90)
print(f"{'Model':<25} {'RMSE':>15} {'MAE':>15} {'R2':>10}")
print("-" * 90)
best_name, best_r2 = None, -np.inf
for name, m in sorted(models.items(), key=lambda kv: kv[1]['r2'], reverse=True):
    print(f"{name:<25} ${m['rmse']:>14,.0f} ${m['mae']:>14,.0f} {m['r2']:>10.4f}")
    if m['r2'] > best_r2:
        best_name, best_r2 = name, m['r2']
print("-" * 90)
print(f"Best model: {best_name} (R2 = {best_r2:.4f})")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE (best model)
# ============================================================================
print(f"\n[STEP 8] Top 15 features ({best_name}):")
best_model = models[best_name]['model']
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
else:
    importances = pd.Series(np.abs(best_model.coef_), index=X_train.columns).sort_values(ascending=False)

for i, (feat, imp) in enumerate(importances.head(15).items(), 1):
    print(f"  {i:2d}. {feat:45s}: {imp:10.6f}")

print("\nPipeline complete.")
