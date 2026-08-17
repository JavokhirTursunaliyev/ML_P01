import pandas as pd
df = pd.read_csv('EXTR_Parcel_processed.csv', encoding='latin-1')
print('='*80)
print('PROCESSED PARCEL DATA - COMPLETE REFERENCE')
print('='*80)
print(f'\\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns')
print('\\nALL 75 COLUMNS WITH DATA TYPES AND STATISTICS:')
print('-'*80)
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    non_null = df[col].notna().sum()
    null_cnt = df[col].isna().sum()
    pct = (null_cnt/len(df))*100
    print(f'{i:2d}. {col:50s} | {str(dtype):15s} | Non-Null: {non_null:7d} | Null: {null_cnt:5d} ({pct:5.1f}%)')
print('\\n' + '='*80)
print('SAMPLE DATA - FIRST 3 ROWS')
print('='*80)
print(df.head(3).to_string())
