import pandas as pd

# Load both files
df_original = pd.read_csv('EXTR_Parcel.csv', encoding='latin-1')
df_processed = pd.read_csv('EXTR_Parcel_processed.csv', encoding='latin-1')

print('='*100)
print('COLUMN TRANSFORMATION SUMMARY')
print('='*100)

# Get column sets
orig_cols = set(df_original.columns)
proc_cols = set(df_processed.columns)

# Identify changes
removed_cols = orig_cols - proc_cols
added_cols = proc_cols - orig_cols
preserved_cols = orig_cols & proc_cols

print(f'\\nOriginal Columns: {len(orig_cols)}')
print(f'Processed Columns: {len(proc_cols)}')
print(f'Preserved (same name): {len(preserved_cols)}')
print(f'Removed Columns: {len(removed_cols)}')
print(f'Added/Created Columns: {len(added_cols)}')

if removed_cols:
    print('\\n' + '-'*100)
    print('REMOVED COLUMNS (were in original, not in processed):')
    print('-'*100)
    for col in sorted(removed_cols):
        print(f'  - {col}')

if added_cols:
    print('\\n' + '-'*100)
    print('ADDED/CREATED COLUMNS (in processed, not in original):')
    print('-'*100)
    for col in sorted(added_cols):
        dtype = df_processed[col].dtype
        print(f'  + {col:50s} | {str(dtype):15s}')

print('\\n' + '='*100)
print('PRESERVED COLUMNS WITH TYPE CHANGES:')
print('='*100)
print('-'*100)
type_changes = []
for col in sorted(preserved_cols):
    orig_type = df_original[col].dtype
    proc_type = df_processed[col].dtype
    if orig_type != proc_type:
        type_changes.append((col, orig_type, proc_type))
        print(f'{col:50s} | {str(orig_type):15s} -> {str(proc_type):15s}')

if not type_changes:
    print('No type changes detected for preserved columns.')
