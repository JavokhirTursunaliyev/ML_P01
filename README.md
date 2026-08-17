# King County House Price Prediction

# Before everything starts files of this codes are not in this repo because uploading limitations.# Reminder
Machine-learning pipelines for predicting King County residential sale prices from parcel, residential building, and real property sales extracts.

The project compares a baseline model with a v3 pipeline that adds domain-driven feature engineering and hyperparameter tuning. It is structured for portfolio review, reproducible local runs, and a clean GitHub upload.

## Repository Contents

| File | Purpose |
| --- | --- |
| `pipeline_v3_fast.py` | Fast v3 pipeline for demos and validation. |
| `pipeline_v3_enhanced.py` | Full v3 pipeline with broader model search. |
| `pipeline_v2.py` | Baseline pipeline for comparison. |
| `analyze.py` | Legacy exploratory pipeline. |
| `analyze_fixed.py` | Processed parcel data reference script. |
| `compare_columns.py` | Compares original and processed parcel columns. |
| `EXECUTIVE_SUMMARY.md` | Short project summary and portfolio framing. |
| `TECHNICAL_ANALYSIS.md` | Deeper methodology notes. |
| `FEATURE_REFERENCE.md` | Engineered feature reference. |
| `DATA.md` | Data setup notes for files excluded from Git. |
| `GITHUB_UPLOAD_CHECKLIST.md` | Commands for creating and pushing the new repo. |

Jupyter notebooks are included for exploration and have been cleared of outputs.

## Data Files

The raw CSV extracts are intentionally excluded from Git because they are large. Place these files in the project root before running the pipelines:

| Required by | File |
| --- | --- |
| v2 and v3 pipelines | `EXTR_Parcel_processed.csv` |
| v2 and v3 pipelines | `EXTR_ResBldg.csv` |
| v2 and v3 pipelines | `EXTR_RPSale.csv` |
| legacy exploration | `EXTR_Parcel.csv` |

See `DATA.md` for file sizes and upload options.

## Quick Start

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline_v3_fast.py
```

For the full run:

```powershell
python pipeline_v3_enhanced.py
```

Expected runtime depends on hardware and dataset availability:

| Script | Typical Use | Runtime |
| --- | --- | --- |
| `pipeline_v3_fast.py` | Fast iteration, demo, smoke test | 5-10 minutes |
| `pipeline_v3_enhanced.py` | Final run and portfolio result | 15-25 minutes |
| `pipeline_v2.py` | Baseline comparison | 10-20 minutes |

## Modeling Approach

The v3 pipelines:

- Load parcel, residential building, and real property sales extracts.
- Filter to market-like sales using `SalePrice > 10000` and empty `SaleWarning`.
- Keep the most recent valid sale per parcel.
- Merge datasets on `Major` and `Minor`.
- Engineer real estate features before modeling.
- Split train/test before fitting `RobustScaler` to avoid leakage.
- Compare tree-based ensemble models and report RMSE, MAE, and R2.

## Engineered Features

The v3 pipeline adds these domain features:

| Feature | Signal |
| --- | --- |
| `HouseAge` | Age of the home at sale time. |
| `IsRenovated` | Whether a renovation year is present. |
| `YearsSinceRenovation` | Recency of renovation, with `999` for never renovated. |
| `LotToBuildingRatio` | Land size relative to living space. |
| `ConditionScore` | Building condition score. |
| `BathroomCount` | Aggregated bathroom-related columns. |
| `BedroomCount` | Aggregated bedroom-related columns. |
| `RecentRenovation` | Renovation within 10 years of sale. |
| `IsNewConstruction` | Home less than 5 years old at sale time. |

See `FEATURE_REFERENCE.md` for the feature rationale and expected importance.

## Expected Results

Baseline v2 reference:

- R2: about 0.4777
- RMSE: about $467,674
- MAE: about $269,717

Expected v3 range:

- R2: about 0.50-0.53
- RMSE: about $430K-$450K
- MAE: about $245K-$255K

Actual results depend on the data snapshot and runtime settings.

## GitHub Upload

This folder is prepared so source code, notebooks, and documentation can be committed without the large CSV extracts. After creating an empty GitHub repository, run:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

Use `GITHUB_UPLOAD_CHECKLIST.md` for the complete flow.

## Notes

- No formal license file has been added. Add one before publishing if you want reuse permissions to be explicit.
- The scripts expect data files in the project root. If you move data into a `data/` directory later, update the read paths consistently.
